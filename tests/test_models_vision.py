"""Real YuNet / SFace checks (skipped until `make models`). No face images are used anywhere."""
import io

import cv2
import numpy as np
import pytest

from attendance.config import load_settings
from attendance.models import ModelIntegrityError, ensure, sha256, verify
from attendance.vision import ARCFACE_TEMPLATE, Face, align, assess
from tests.conftest import needs_models


def test_alignment_maps_landmarks_onto_template():
    img = np.zeros((300, 300, 3), np.uint8)
    lm = np.array([[120, 130], [180, 128], [151, 165], [126, 200], [176, 198]], np.float32)
    for x, y in lm:
        cv2.circle(img, (int(x), int(y)), 3, (255, 255, 255), -1)
    crop = align(img, lm)
    assert crop.shape == (112, 112, 3)
    ys, xs = np.nonzero(crop[:, :, 0] > 120)
    assert abs(xs.mean() - ARCFACE_TEMPLATE[:, 0].mean()) < 3 and abs(ys.mean() - ARCFACE_TEMPLATE[:, 1].mean()) < 3


def test_quality_gates():
    q = load_settings().quality
    rng = np.random.default_rng(0)
    sharp = rng.integers(0, 256, (200, 200, 3), dtype=np.uint8)
    lm = np.array([[70, 80], [130, 80], [100, 110], [75, 140], [125, 140]], np.float32)
    face = Face(box=(40, 40, 120, 120), landmarks=lm, score=0.99)
    assert assess(sharp, face, q).ok
    blurry = cv2.GaussianBlur(np.full((200, 200, 3), 128, np.uint8), (0, 0), 5)
    assert "blurry" in assess(blurry, face, q).reasons
    small = Face(box=(40, 40, 30, 30), landmarks=lm, score=0.9)
    assert "face_too_small" in assess(sharp, small, q).reasons
    turned = Face(box=face.box, landmarks=lm + np.array([[0, 0], [0, 0], [40, 0], [0, 0], [0, 0]], np.float32),
                  score=0.9)
    assert "not_frontal" in assess(sharp, turned, q).reasons
    tilted = lm.copy()
    tilted[1] = [130, 130]
    assert "head_tilted" in assess(sharp, Face(face.box, tilted, 0.9), q).reasons
    dark = (sharp // 8).astype(np.uint8)
    assert "too_dark" in assess(dark, face, q).reasons


def test_model_download_is_verified(tmp_path):
    s = load_settings(env={"ATTENDANCE_MODEL_DIR": str(tmp_path)})

    class Resp(io.BytesIO):
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

    with pytest.raises(ModelIntegrityError):
        ensure(s, opener=lambda url: Resp(b"not the model"))
    assert not list(tmp_path.iterdir())              # the bad download was removed, nothing half-written


@needs_models
def test_pinned_hashes():
    s = load_settings()
    for which in ("detector", "recognizer"):
        verify(s.model_path(which), getattr(s, which))
        assert sha256(s.model_path(which)) == getattr(s, which).sha256


@needs_models
def test_no_faces_in_faceless_frames_and_embedding_contract():
    from attendance.vision import FaceEngine

    s = load_settings()
    eng = FaceEngine(s)
    rng = np.random.default_rng(1)
    for frame in (np.full((480, 640, 3), 128, np.uint8), rng.integers(0, 256, (480, 640, 3), dtype=np.uint8)):
        assert eng.detector.detect(frame) == []
        assert eng.analyze(frame).status == "no_face"
    crop = cv2.GaussianBlur(rng.integers(0, 256, (112, 112, 3), dtype=np.uint8), (0, 0), 2)
    v = eng.embedder.embed(crop)
    assert v.shape == (128,) and abs(float(np.linalg.norm(v)) - 1) < 1e-5
    ref = cv2.FaceRecognizerSF.create(str(s.model_path("recognizer")), "").feature(crop).ravel()
    assert float(v @ (ref / np.linalg.norm(ref))) > 0.9999            # ONNX Runtime == OpenCV's own SFace
    assert eng.analyze(b"not an image").status == "bad_image"
    with pytest.raises(ValueError):
        eng.embedder.embed(np.zeros((100, 100, 3), np.uint8))


@needs_models
def test_alignment_matches_opencv_aligncrop():
    s = load_settings()
    rng = np.random.default_rng(2)
    img = cv2.GaussianBlur(rng.integers(0, 256, (300, 300, 3), dtype=np.uint8), (0, 0), 3)
    lm = np.array([[120, 130], [180, 128], [151, 165], [126, 200], [176, 198]], np.float32)
    ref = cv2.FaceRecognizerSF.create(str(s.model_path("recognizer")), "").alignCrop(
        img, np.array([100, 90, 110, 140, *lm.ravel(), 0.99], np.float32))
    assert np.abs(align(img, lm).astype(int) - ref.astype(int)).mean() < 1.0


@needs_models
def test_image_endpoint_with_a_faceless_frame(client):
    from tests.conftest import D

    ok, buf = cv2.imencode(".jpg", np.full((240, 320, 3), 90, np.uint8))
    r = client.post("/v1/recognize", files={"image": ("f.jpg", buf.tobytes(), "image/jpeg")},
                    data={"device_id": "gate"}, headers=D)
    assert r.status_code == 200 and r.json()["decision"] == "no_face"
    assert set(r.json()["timings_ms"]) == {"decode", "detect"}
