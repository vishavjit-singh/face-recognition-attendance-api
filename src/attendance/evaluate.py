"""Accuracy of the real pipeline (YuNet → alignment → SFace) on the Olivetti research faces.

The Olivetti set (AT&T Laboratories Cambridge, 40 subjects × 10 greyscale 64×64 images) is downloaded
by scikit-learn on first use. That download is blocked in the environment this project was built
in, so these numbers are produced by the CI `evaluate` job and published as a build artifact.

Measured:
* detection rate (Olivetti crops are tight, so each image is upscaled and padded first; images where
  YuNet finds no face fall back to a plain resize to 112×112, and that count is reported)
* verification: ROC AUC, EER, TAR at FAR = 1% and 0.1%, and TAR/FAR at the configured threshold
* open-set identification: 30 enrolled subjects (3 templates each), probes = their other 7 images
  plus all images of the 10 non-enrolled subjects; rank-1 accuracy, correct-accept and false-accept
  rates with the configured threshold and margin
"""
from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np

from .config import Settings
from .matching import Gallery


def load_olivetti():
    from sklearn.datasets import fetch_olivetti_faces

    d = fetch_olivetti_faces(shuffle=False)
    return d.images, d.target


def embed_all(settings: Settings, images: np.ndarray) -> tuple[np.ndarray, int]:
    from .vision import FaceEngine, align

    engine = FaceEngine(settings)
    vecs, detected = [], 0
    for img in images:
        g = (np.clip(img, 0, 1) * 255).astype(np.uint8)
        big = cv2.resize(g, (192, 192), interpolation=cv2.INTER_CUBIC)
        big = cv2.copyMakeBorder(big, 64, 64, 64, 64, cv2.BORDER_CONSTANT, value=int(np.median(g)))
        bgr = cv2.cvtColor(big, cv2.COLOR_GRAY2BGR)
        faces = engine.detector.detect(bgr)
        if faces:
            detected += 1
            crop = align(bgr, faces[0].landmarks)
        else:
            crop = cv2.cvtColor(cv2.resize(g, (112, 112), interpolation=cv2.INTER_CUBIC), cv2.COLOR_GRAY2BGR)
        vecs.append(engine.embedder.embed(crop))
    return np.stack(vecs), detected


def verification(emb: np.ndarray, labels: np.ndarray, threshold: float) -> dict:
    from sklearn.metrics import roc_auc_score, roc_curve

    sims = emb @ emb.T
    iu = np.triu_indices(len(labels), k=1)
    scores = sims[iu]
    same = (labels[:, None] == labels[None, :])[iu]
    fpr, tpr, thr = roc_curve(same, scores)
    eer_i = int(np.argmin(np.abs(fpr - (1 - tpr))))

    def tar_at(far):
        ok = np.where(fpr <= far)[0]
        i = ok[-1]
        return float(tpr[i]), float(thr[i])

    tar1, thr1 = tar_at(1e-2)
    tar01, thr01 = tar_at(1e-3)
    return {"genuine_pairs": int(same.sum()), "impostor_pairs": int((~same).sum()),
            "roc_auc": round(float(roc_auc_score(same, scores)), 4), "eer": round(float(fpr[eer_i]), 4),
            "tar_at_far_1e-2": round(tar1, 4), "threshold_at_far_1e-2": round(thr1, 4),
            "tar_at_far_1e-3": round(tar01, 4), "threshold_at_far_1e-3": round(thr01, 4),
            "at_configured_threshold": {"threshold": threshold,
                                        "tar": round(float((scores[same] >= threshold).mean()), 4),
                                        "far": round(float((scores[~same] >= threshold).mean()), 5)},
            "_curve": {"fpr": fpr.tolist(), "tpr": tpr.tolist()},
            "_scores": {"genuine": scores[same].tolist(), "impostor": scores[~same].tolist()}}


def identification(emb: np.ndarray, labels: np.ndarray, settings: Settings, enrolled: int = 30,
                   templates: int = 3) -> dict:
    subjects = sorted(set(labels.tolist()))
    enrolled = min(enrolled, max(1, int(len(subjects) * 0.75)))
    known = set(subjects[:enrolled])
    g_idx, probes = [], []
    seen: dict[int, int] = {}
    for i, lab in enumerate(labels.tolist()):
        if lab in known and seen.get(lab, 0) < templates:
            g_idx.append(i)
            seen[lab] = seen.get(lab, 0) + 1
        else:
            probes.append(i)
    gallery = Gallery(emb[g_idx], labels[g_idx].astype(np.int64))
    rank1 = correct_accept = false_accept = known_n = unknown_n = 0
    for i in probes:
        lab = int(labels[i])
        ranked = gallery.scores(emb[i])
        m = gallery.identify(emb[i], settings.matching)
        if lab in known:
            known_n += 1
            rank1 += ranked[0][0] == lab
            correct_accept += m.decision == "accepted" and m.person_id == lab
        else:
            unknown_n += 1
            false_accept += m.decision == "accepted"
    return {"enrolled_subjects": enrolled, "templates_per_subject": templates, "known_probes": known_n,
            "unknown_probes": unknown_n, "rank1_closed_set": round(rank1 / max(known_n, 1), 4),
            "correct_accept_rate": round(correct_accept / max(known_n, 1), 4),
            "false_accept_rate_unknowns": round(false_accept / max(unknown_n, 1), 4)}


def run(settings: Settings, out_dir: Path, loader=load_olivetti) -> dict:
    images, labels = loader()
    emb, detected = embed_all(settings, images)
    ver = verification(emb, labels, settings.matching.threshold)
    result = {"dataset": "Olivetti faces (AT&T Laboratories Cambridge), via scikit-learn",
              "images": int(len(labels)), "subjects": int(len(set(labels.tolist()))),
              "detected": detected, "detection_rate": round(detected / len(labels), 4),
              "verification": {k: v for k, v in ver.items() if not k.startswith("_")},
              "identification": identification(emb, labels, settings)}
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "evaluation.json").write_text(json.dumps(result, indent=2))
    _plot(ver, settings.matching.threshold, out_dir / "evaluation_roc.png")
    return result


def _plot(ver: dict, threshold: float, path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 4.4))
    a1.plot(ver["_curve"]["fpr"], ver["_curve"]["tpr"], color="#2a78d6", lw=2)
    a1.set_xscale("log")
    a1.set_xlim(1e-4, 1)
    a1.set_xlabel("false accept rate")
    a1.set_ylabel("true accept rate")
    a1.set_title(f"Verification ROC (AUC {ver['roc_auc']})", loc="left", weight="bold")
    a1.grid(color="#e3e7ee")
    bins = np.linspace(-0.3, 1, 66)
    a2.hist(ver["_scores"]["impostor"], bins=bins, color="#e34948", alpha=0.6, density=True, label="impostor")
    a2.hist(ver["_scores"]["genuine"], bins=bins, color="#1baf7a", alpha=0.6, density=True, label="genuine")
    a2.axvline(threshold, color="#0e1a2b", ls="--", lw=1.2, label=f"threshold {threshold}")
    a2.set_xlabel("cosine similarity")
    a2.legend(frameon=False)
    a2.set_title("Score distributions (Olivetti)", loc="left", weight="bold")
    fig.tight_layout()
    fig.savefig(path, dpi=150, facecolor="white")
    plt.close(fig)
