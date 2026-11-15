"""`make demo`: one simulated working day through the real API, plus a check of the real models.

1. **Model plumbing (real YuNet + SFace).** Load and verify both models, check that frames with no
   face (blank, noise, gradient) give zero detections, that our alignment matches OpenCV's
   ``alignCrop`` and that SFace returns a unit 128-d vector; time each stage.
2. **A day of attendance (SYNTHETIC embeddings).** 24 fictional employees are enrolled with three
   templates each. Then a camera at the entrance reports sightings from 08:40 to 19:10: arrivals and
   departures seen in several consecutive frames, a near-twin pair, poor captures, strangers and two
   frames with no face (real images through the real detector). Everything goes through the HTTP API
   in-process, with an admin and a device API key.
"""
from __future__ import annotations

import json
import statistics
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import cv2
import numpy as np

from .config import ROOT, Settings
from .synthetic import Identities

DAY = "2026-11-16"
IST = timedelta(minutes=330)
N_PEOPLE = 24
LOOKALIKES = ((6, 7),)
ADMIN, DEVICE = "demo-admin-key", "demo-device-key"


def utc(hhmm: str, seconds: float = 0) -> str:
    h, m = map(int, hhmm.split(":"))
    t = datetime.fromisoformat(f"{DAY}T00:00:00+00:00") + timedelta(hours=h, minutes=m, seconds=seconds) - IST
    return t.astimezone(timezone.utc).isoformat(timespec="seconds")


def plumbing(settings: Settings, runs: int = 15) -> dict:
    from .vision import FaceEngine, align

    t0 = time.perf_counter()
    engine = FaceEngine(settings)
    load_ms = (time.perf_counter() - t0) * 1000
    rng = np.random.default_rng(3)
    frames = {"blank": np.full((480, 640, 3), 128, np.uint8),
              "noise": rng.integers(0, 256, (480, 640, 3), dtype=np.uint8),
              "gradient": np.tile(np.linspace(0, 255, 640, dtype=np.uint8), (480, 1))[:, :, None].repeat(3, 2)}
    detections = {k: len(engine.detector.detect(v)) for k, v in frames.items()}
    detect_ms = []
    for _ in range(runs):
        t0 = time.perf_counter()
        engine.detector.detect(frames["noise"])
        detect_ms.append((time.perf_counter() - t0) * 1000)
    # alignment vs OpenCV's own alignCrop, on a textured image with plausible landmarks
    img = cv2.GaussianBlur(rng.integers(0, 256, (300, 300, 3), dtype=np.uint8), (0, 0), 3)
    lm = np.array([[120, 130], [180, 128], [151, 165], [126, 200], [176, 198]], np.float32)
    ours = align(img, lm)
    sf = cv2.FaceRecognizerSF.create(str(settings.model_path("recognizer")), "")
    ref = sf.alignCrop(img, np.array([100, 90, 110, 140, *lm.ravel(), 0.99], np.float32))
    align_diff = float(np.abs(ours.astype(np.int16) - ref.astype(np.int16)).mean())
    embed_ms, vec = [], None
    for _ in range(runs):
        t0 = time.perf_counter()
        vec = engine.embedder.embed(ours)
        embed_ms.append((time.perf_counter() - t0) * 1000)
    cv_vec = sf.feature(ours).ravel()
    cv_vec /= np.linalg.norm(cv_vec)
    return {"models_load_ms": round(load_ms, 1), "detections_on_faceless_frames": detections,
            "detect_ms_median_640x480": round(statistics.median(detect_ms), 2),
            "embed_ms_median": round(statistics.median(embed_ms), 2),
            "embedding_dim": int(vec.size), "embedding_norm": round(float(np.linalg.norm(vec)), 6),
            "cosine_vs_opencv_feature": round(float(vec @ cv_vec), 6),
            "alignment_mean_abs_diff_vs_opencv": round(align_diff, 3)}


def simulate(client, rng: np.random.Generator) -> dict:
    """Enrol and run the day through the API. Returns ground truth for checking the report."""
    ids = Identities(N_PEOPLE, seed=9, lookalikes=LOOKALIKES)
    A, D = {"X-API-Key": ADMIN}, {"X-API-Key": DEVICE}
    pid = {}
    guard_hits: list[dict] = []
    for i in range(N_PEOPLE):
        r = client.post("/v1/people", json={"external_id": f"E{i + 1:03d}", "name": f"Employee {i + 1:02d}"},
                        headers=A)
        pid[i] = r.json()["id"]
        for _ in range(3):
            body = {"embedding": ids.capture(i, 0.5).tolist()}
            r = client.post(f"/v1/people/{pid[i]}/templates", json=body, headers=A)
            if r.status_code == 409 and r.json()["error"]["code"] == "matches_other_person":
                guard_hits.append(r.json()["error"])            # near-twin: an admin confirms, audited
                r = client.post(f"/v1/people/{pid[i]}/templates", json={**body, "force": True}, headers=A)
            assert r.status_code == 201, r.text
    absent = {4, 15}
    no_checkout = {19}
    sightings: list[tuple[str, dict, dict]] = []      # (ts, payload, truth)

    def seen(i: int | None, ts_hhmm: str, sec: float, sigma: float, kind: str):
        vec = ids.stranger() if i is None else ids.capture(i, sigma)
        sightings.append((utc(ts_hhmm, sec), {"embedding": vec.tolist()}, {"who": i, "kind": kind}))

    for i in range(N_PEOPLE):
        if i in absent:
            continue
        arrive = 9 * 60 + 20 + rng.normal(0, 12) + (25 if i in (2, 11, 21) else 0)
        hh, mm = divmod(int(arrive), 60)
        frames = int(rng.integers(2, 5))
        for f in range(frames):                       # the same walk-in seen in several frames
            seen(i, f"{hh:02d}:{mm:02d}", f * 4.0, float(rng.uniform(0.6, 1.0)), "arrival")
        if i % 5 == 0:                                # walks past the camera at lunch
            seen(i, "13:" + f"{int(rng.integers(0, 50)):02d}", 0, float(rng.uniform(0.6, 1.0)), "midday")
        if i in no_checkout:
            continue
        leave = 17 * 60 + 40 + rng.normal(0, 30)
        hh, mm = divmod(int(leave), 60)
        for f in range(int(rng.integers(1, 3))):
            seen(i, f"{hh:02d}:{mm:02d}", f * 5.0, float(rng.uniform(0.6, 1.0)), "departure")
    for k in range(4):                                # poor captures (motion blur, angle): high noise
        i = int(rng.choice([j for j in range(N_PEOPLE) if j not in absent]))
        seen(i, f"12:{10 + k * 7:02d}", 0, 2.6, "poor_capture")
    for k in range(6):                                # visitors who are not enrolled
        seen(None, f"{10 + k}:15", 0, 0, "stranger")
    for k in range(5):                                # the near-twin pair, noisy captures
        seen(7, f"15:{10 + k * 5:02d}", 0, 1.05, "lookalike")
    sightings.sort(key=lambda s: s[0])
    results = []
    for ts, payload, truth in sightings:
        r = client.post("/v1/recognize", json={**payload, "device_id": "gate-1", "ts": ts}, headers=D)
        assert r.status_code == 200, r.text
        results.append({"ts": ts, **truth, "who_id": None if truth["who"] is None else pid[truth["who"]],
                        **r.json()})
    for name, frame in (("blank", np.full((480, 640, 3), 128, np.uint8)),
                        ("noise", rng.integers(0, 256, (480, 640, 3), dtype=np.uint8))):
        ok, buf = cv2.imencode(".jpg", frame)
        r = client.post("/v1/recognize", files={"image": (f"{name}.jpg", buf.tobytes(), "image/jpeg")},
                        data={"device_id": "gate-1", "ts": utc("12:00")}, headers=D)
        results.append({"ts": utc("12:00"), "who": None, "kind": f"frame_{name}", **r.json()})
    report = client.get(f"/v1/attendance/{DAY}", headers=A).json()
    # right to erasure: delete one person at the end and show what went
    erased = client.request("DELETE", f"/v1/people/{pid[23]}", headers=A).json()
    return {"pid": pid, "guard_hits": guard_hits, "absent": sorted(absent), "no_checkout": sorted(no_checkout),
            "results": results,
            "report": report, "erasure": erased,
            "audit": client.get("/v1/admin/audit", headers=A).json()["audit"][:3],
            "identities": ids}


def similarity_samples(ids: Identities, rng: np.random.Generator, n: int = 3000) -> dict:
    """Scores the matcher would see: genuine, impostor and near-twin pairs at demo capture quality."""
    gen, imp, twin = [], [], []
    for _ in range(n):
        a = int(rng.integers(0, N_PEOPLE))
        b = int((a + rng.integers(1, N_PEOPLE)) % N_PEOPLE)
        s1, s2 = 0.5, float(rng.uniform(0.6, 1.0))
        gen.append(float(ids.capture(a, s1) @ ids.capture(a, s2)))
        if {a, b} != {6, 7}:
            imp.append(float(ids.capture(a, s1) @ ids.capture(b, s2)))
    for _ in range(n // 6):
        twin.append(float(ids.capture(6, 0.5) @ ids.capture(7, float(rng.uniform(0.6, 1.0)))))
    return {"genuine": gen, "impostor": imp, "near_twin": twin}


def run(settings: Settings, out_dir: Path | None = None, with_models: bool = True) -> dict:
    from starlette.testclient import TestClient

    from .api import create_app
    from .store import Store

    out_dir = out_dir or ROOT / "data"
    out_dir.mkdir(parents=True, exist_ok=True)
    db = out_dir / "demo.db"
    for p in (db, db.with_suffix(".db-wal"), db.with_suffix(".db-shm")):
        if p.exists():
            p.unlink()
    settings = Settings(**{**settings.__dict__, "api_keys": {ADMIN: "admin", DEVICE: "device"}, "database": db})
    rng = np.random.default_rng(16)
    plumb = plumbing(settings) if with_models else None
    app = create_app(settings, Store(db))
    with TestClient(app) as client:
        sim = simulate(client, rng)
    sims = similarity_samples(sim.pop("identities"), np.random.default_rng(5))
    out = {"plumbing": plumb, **sim, "similarity": sims, "threshold": settings.matching.threshold,
           "margin": settings.matching.margin}
    (out_dir / "demo.json").write_text(json.dumps(out, indent=1, default=str))
    return out
