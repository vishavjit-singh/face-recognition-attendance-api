"""Detection (YuNet), quality gates, 5-point alignment and SFace embeddings.

* YuNet runs through OpenCV's ``cv2.FaceDetectorYN`` (documented, stable API).
* SFace runs through **ONNX Runtime** directly; the input is the 112×112 aligned RGB crop as float
  NCHW, the output a 128-d vector that we L2-normalise (cosine similarity = dot product).
* Alignment maps the 5 landmarks onto the standard ArcFace 112×112 template with a similarity
  transform, the same template OpenCV's ``FaceRecognizerSF.alignCrop`` uses.
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass, field

import cv2
import numpy as np

from .config import Quality, Settings
from .models import verify

ARCFACE_TEMPLATE = np.array([[38.2946, 51.6963], [73.5318, 51.5014], [56.0252, 71.7366],
                             [41.5493, 92.3655], [70.7299, 92.2041]], dtype=np.float32)


@dataclass
class Face:
    box: tuple[float, float, float, float]          # x, y, w, h
    landmarks: np.ndarray                            # 5×2: right eye, left eye, nose, right mouth, left mouth
    score: float

    @property
    def row(self) -> np.ndarray:
        return np.array([*self.box, *self.landmarks.ravel(), self.score], dtype=np.float32)


@dataclass
class QualityReport:
    face_px: float
    sharpness: float
    brightness: float
    yaw: float
    roll_deg: float
    reasons: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.reasons

    def as_dict(self) -> dict:
        return {"face_px": round(self.face_px, 1), "sharpness": round(self.sharpness, 1),
                "brightness": round(self.brightness, 1), "yaw": round(self.yaw, 3),
                "roll_deg": round(self.roll_deg, 1), "reasons": self.reasons}


def assess(image: np.ndarray, face: Face, q: Quality) -> QualityReport:
    x, y, w, h = face.box
    H, W = image.shape[:2]
    x0, y0 = max(0, int(x)), max(0, int(y))
    x1, y1 = min(W, int(x + w)), min(H, int(y + h))
    region = image[y0:y1, x0:x1]
    grey = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY) if region.ndim == 3 and region.size else region
    sharp = float(cv2.Laplacian(grey, cv2.CV_64F).var()) if grey.size else 0.0
    bright = float(grey.mean()) if grey.size else 0.0
    re, le, nose = face.landmarks[0], face.landmarks[1], face.landmarks[2]
    eye_dist = float(np.linalg.norm(le - re)) or 1e-6
    yaw = float((nose[0] - (re[0] + le[0]) / 2) / eye_dist)
    roll = math.degrees(math.atan2(le[1] - re[1], le[0] - re[0]))
    rep = QualityReport(face_px=float(min(w, h)), sharpness=sharp, brightness=bright, yaw=yaw, roll_deg=roll)
    if rep.face_px < q.min_face_px:
        rep.reasons.append("face_too_small")
    if sharp < q.min_sharpness:
        rep.reasons.append("blurry")
    if not q.brightness[0] <= bright <= q.brightness[1]:
        rep.reasons.append("too_dark" if bright < q.brightness[0] else "too_bright")
    if abs(yaw) > q.max_yaw:
        rep.reasons.append("not_frontal")
    if abs(roll) > q.max_roll_deg:
        rep.reasons.append("head_tilted")
    return rep


def align(image: np.ndarray, landmarks: np.ndarray, size: int = 112) -> np.ndarray:
    """Similarity-transform the 5 landmarks onto the ArcFace template and crop 112×112."""
    m, _ = cv2.estimateAffinePartial2D(landmarks.astype(np.float32), ARCFACE_TEMPLATE, method=cv2.LMEDS)
    if m is None:
        raise ValueError("could not estimate an alignment from the landmarks")
    return cv2.warpAffine(image, m, (size, size), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)


class Detector:
    def __init__(self, settings: Settings):
        path = settings.model_path("detector")
        verify(path, settings.detector)
        self.max_side = settings.max_side
        self.net = cv2.FaceDetectorYN.create(str(path), "", (320, 320), settings.score_threshold,
                                             settings.nms_threshold, 5000)

    def detect(self, image: np.ndarray) -> list[Face]:
        H, W = image.shape[:2]
        scale = min(1.0, self.max_side / max(H, W))
        img = cv2.resize(image, (int(W * scale), int(H * scale))) if scale < 1 else image
        self.net.setInputSize((img.shape[1], img.shape[0]))
        _, faces = self.net.detect(img)
        out = []
        for f in faces if faces is not None else []:
            box = tuple(float(v) / scale for v in f[:4])
            out.append(Face(box=box, landmarks=(f[4:14].reshape(5, 2) / scale).astype(np.float32),
                            score=float(f[14])))
        return sorted(out, key=lambda fc: fc.box[2] * fc.box[3], reverse=True)


class Embedder:
    def __init__(self, settings: Settings):
        import onnxruntime as ort

        path = settings.model_path("recognizer")
        verify(path, settings.recognizer)
        opts = ort.SessionOptions()
        opts.log_severity_level = 3
        self.session = ort.InferenceSession(str(path), opts, providers=["CPUExecutionProvider"])
        self.input = self.session.get_inputs()[0].name

    def embed(self, aligned_bgr: np.ndarray) -> np.ndarray:
        if aligned_bgr.shape[:2] != (112, 112):
            raise ValueError("expected a 112×112 aligned crop")
        x = aligned_bgr[:, :, ::-1].transpose(2, 0, 1)[None].astype(np.float32)
        v = self.session.run(None, {self.input: x})[0].ravel()
        return (v / (np.linalg.norm(v) or 1.0)).astype(np.float32)


@dataclass
class Analysis:
    status: str                      # ok | no_face | multiple_faces | low_quality | bad_image
    faces: int
    quality: QualityReport | None = None
    embedding: np.ndarray | None = None
    timings_ms: dict = field(default_factory=dict)


class FaceEngine:
    """decode → detect → quality → align → embed, with per-stage timings."""

    def __init__(self, settings: Settings):
        self.settings = settings
        self.detector = Detector(settings)
        self.embedder = Embedder(settings)

    def analyze(self, data: bytes | np.ndarray, require_single: bool = False) -> Analysis:
        t = {}
        t0 = time.perf_counter()
        img = data if isinstance(data, np.ndarray) else cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
        t["decode"] = (time.perf_counter() - t0) * 1000
        if img is None or img.size == 0:
            return Analysis("bad_image", 0, timings_ms=t)
        if img.ndim == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        t0 = time.perf_counter()
        faces = self.detector.detect(img)
        t["detect"] = (time.perf_counter() - t0) * 1000
        if not faces:
            return Analysis("no_face", 0, timings_ms=t)
        if require_single and len(faces) > 1:
            return Analysis("multiple_faces", len(faces), timings_ms=t)
        face = faces[0]
        t0 = time.perf_counter()
        q = assess(img, face, self.settings.quality)
        t["quality"] = (time.perf_counter() - t0) * 1000
        if not q.ok:
            return Analysis("low_quality", len(faces), quality=q, timings_ms=t)
        t0 = time.perf_counter()
        crop = align(img, face.landmarks)
        t["align"] = (time.perf_counter() - t0) * 1000
        t0 = time.perf_counter()
        emb = self.embedder.embed(crop)
        t["embed"] = (time.perf_counter() - t0) * 1000
        return Analysis("ok", len(faces), quality=q, embedding=emb, timings_ms=t)
