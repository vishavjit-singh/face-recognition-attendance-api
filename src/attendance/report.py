"""README images and summary numbers, computed from the demo run (data/demo.json)."""
from __future__ import annotations

import json
from collections import Counter
from datetime import datetime, timedelta
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Patch  # noqa: E402

INK, MUTED, GRID = "#0e1a2b", "#5b6676", "#e3e7ee"
ACCENT, SIGNAL, BAD, GOOD, PURPLE = "#2a78d6", "#eda100", "#e34948", "#1baf7a", "#6250d6"
IST = timedelta(minutes=330)

plt.rcParams.update({
    "font.family": "DejaVu Sans", "axes.edgecolor": GRID, "axes.labelcolor": MUTED, "xtick.color": MUTED,
    "ytick.color": MUTED, "axes.titleweight": "bold", "axes.titlesize": 13, "axes.titlecolor": INK,
    "axes.titlelocation": "left", "axes.spines.top": False, "axes.spines.right": False, "figure.dpi": 150,
})


def _save(fig, out: Path, name: str) -> Path:
    fig.tight_layout()
    p = out / name
    fig.savefig(p, facecolor="white")
    plt.close(fig)
    return p


def summary(demo: dict) -> dict:
    res = demo["results"]
    rep = demo["report"]["summary"]
    return {
        "plumbing": demo["plumbing"],
        "enrolled": rep["enrolled"],
        "near_twin_enrolment_guard_hits": len(demo["guard_hits"]),
        "sightings": len(res),
        "decisions": dict(Counter(r["decision"] for r in res)),
        "events": dict(Counter(r.get("event") for r in res if r["decision"] == "accepted")),
        "wrong_person_accepted": sum(1 for r in res if r["decision"] == "accepted" and r.get("who_id") is not None
                                     and r["person_id"] != r["who_id"]),
        "strangers_accepted": sum(1 for r in res if r["kind"] == "stranger" and r["decision"] == "accepted"),
        "poor_captures": dict(Counter(r["decision"] for r in res if r["kind"] == "poor_capture")),
        "report": {k: rep[k] for k in ("on_time", "late", "absent")},
        "missing_check_out": sum(1 for p in demo["report"]["people"] if p["missing_check_out"]),
        "erasure": demo["erasure"],
    }


def architecture(out: Path) -> Path:
    fig, ax = plt.subplots(figsize=(13, 5.8))
    ax.set_xlim(0, 13)
    ax.set_ylim(0, 6.2)
    ax.axis("off")

    def box(x, y, w, h, title, sub, color=INK):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.12", fc="white",
                                    ec=color, lw=2))
        ax.text(x + w / 2, y + h * 0.7, title, ha="center", va="center", fontsize=11, weight="bold", color=color)
        ax.text(x + w / 2, y + h * 0.32, sub, ha="center", va="center", fontsize=8.2, color=MUTED, linespacing=1.4)

    def arrow(x1, y1, x2, y2, color=MUTED):
        ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=14, lw=1.6, color=color))

    ax.text(0.15, 5.9, "face-recognition-attendance-api: frames in, embeddings only, attendance out",
            fontsize=13.5, weight="bold", color=INK)
    box(0.2, 3.4, 2.0, 1.4, "Gate camera", "frame (JPEG)\nor an embedding from\nan edge device")
    box(2.7, 3.4, 2.2, 1.4, "YuNet detector", "cv2.FaceDetectorYN\nbox + 5 landmarks", PURPLE)
    box(5.3, 3.4, 2.2, 1.4, "Quality gates", "size · blur · brightness\nyaw · roll", SIGNAL)
    box(7.9, 3.4, 2.2, 1.4, "Align + SFace", "ArcFace 5-pt template\nONNX Runtime → 128-d", PURPLE)
    box(10.5, 3.4, 2.3, 1.4, "Open-set match", "cosine ≥ threshold\nmargin vs runner-up\nelse unknown/ambiguous",
        ACCENT)
    box(10.5, 1.2, 2.3, 1.5, "Attendance rules",
        "first sighting = check-in\ncooldown dedupe\nlast sighting = check-out", GOOD)
    box(7.6, 1.2, 2.5, 1.5, "SQLite", "people · templates (embeddings)\nevents · audit\nno images stored", INK)
    box(4.7, 1.2, 2.5, 1.5, "Starlette API",
        "X-API-Key: admin / device\nenrol · recognise · report\nerase · purge", ACCENT)
    box(0.2, 1.2, 4.1, 1.5, "Privacy controls", "right to erasure (person + templates + events)\n"
        "retention purge (90 d events, 7 d unknowns)\nnear-twin enrolment guard · audit log", BAD)
    arrow(2.2, 4.1, 2.7, 4.1)
    arrow(4.9, 4.1, 5.3, 4.1)
    arrow(7.5, 4.1, 7.9, 4.1)
    arrow(10.1, 4.1, 10.5, 4.1)
    arrow(11.65, 3.4, 11.65, 2.7)
    arrow(10.5, 1.95, 10.1, 1.95)
    arrow(7.6, 1.95, 7.2, 1.95)
    arrow(4.7, 1.95, 4.3, 1.95)
    ax.text(5.3, 3.05, "rejected → no_face / low_quality (no embedding computed)", fontsize=8, color="#9a6a00")
    ax.text(0.2, 0.55, "Models: OpenCV Zoo YuNet (2023mar) and SFace (2021dec), downloaded by `make models` and "
            "checked against pinned SHA-256 before loading.", fontsize=8.5, color=MUTED)
    return _save(fig, out, "architecture.png")


def similarity(demo: dict, out: Path) -> Path:
    s = demo["similarity"]
    thr, margin = demo["threshold"], demo["margin"]
    fig, ax = plt.subplots(figsize=(11, 4.2))
    bins = np.linspace(-0.4, 1.0, 71)
    ax.hist(s["impostor"], bins=bins, color=BAD, alpha=0.55, density=True, label="impostor pairs")
    ax.hist(s["near_twin"], bins=bins, color=SIGNAL, alpha=0.7, density=True, label="near-twin pair (E007/E008)")
    ax.hist(s["genuine"], bins=bins, color=GOOD, alpha=0.55, density=True, label="genuine pairs")
    ax.axvline(thr, color=INK, ls="--", lw=1.3)
    ax.text(thr + 0.01, ax.get_ylim()[1] * 0.92, f"threshold {thr}\n(+ margin {margin} over the runner-up)",
            fontsize=8.5, color=INK, va="top")
    ax.set_xlabel("cosine similarity")
    ax.set_yticks([])
    ax.legend(frameon=False, fontsize=9, loc="upper left")
    ax.set_title("Similarity scores in the demo: SYNTHETIC embeddings (not real faces, see evaluate for SFace)")
    return _save(fig, out, "similarity.png")


def timeline(demo: dict, out: Path) -> Path:
    people = demo["report"]["people"]
    res = demo["results"]

    def hrs(ts: str) -> float:
        t = datetime.fromisoformat(ts) + IST
        return t.hour + t.minute / 60 + t.second / 3600

    fig, ax = plt.subplots(figsize=(12, 7.2))
    names = [p["external_id"] for p in people]
    y = {p["person_id"]: i for i, p in enumerate(people)}
    for p in people:
        i = y[p["person_id"]]
        if p["status"] == "absent":
            ax.text(9.5, i, "absent", va="center", fontsize=8, color=MUTED)
            continue
        cin = int(p["check_in"][:2]) + int(p["check_in"][3:]) / 60
        cout = int(p["check_out"][:2]) + int(p["check_out"][3:]) / 60 if p["check_out"] else 19.25
        color = SIGNAL if p["status"] == "late" else GOOD
        ax.barh(i, cout - cin, left=cin, height=0.55, color=color, alpha=0.85 if p["check_out"] else 0.35,
                hatch=None if p["check_out"] else "//", edgecolor="white")
        if not p["check_out"]:
            ax.text(cout - 0.1, i, "no check-out", ha="right", va="center", fontsize=7.5, color=INK)
    for r in res:
        if r["decision"] == "accepted" and r.get("person_id") in y:
            ax.plot(hrs(r["ts"]), y[r["person_id"]], "|", color=INK if r.get("event") != "duplicate" else MUTED,
                    ms=8 if r.get("event") != "duplicate" else 5, alpha=0.9)
    rug = len(people) + 0.4
    for r in res:
        if r["decision"] in ("unknown", "ambiguous", "no_face"):
            c = {"unknown": BAD, "ambiguous": PURPLE, "no_face": MUTED}[r["decision"]]
            ax.plot(hrs(r["ts"]), rug, "v", color=c, ms=7)
    ax.axvline(9.5 + 10 / 60, color=SIGNAL, ls=":", lw=1.2)
    ax.text(9.5 + 10 / 60 + 0.05, -1.1, "late after 09:40", fontsize=8, color="#9a6a00")
    ax.set_yticks(list(range(len(people))) + [rug], names + ["not matched"], fontsize=8)
    ax.set_ylim(len(people) + 1.2, -1.6)
    ax.set_xlim(8.5, 19.5)
    ax.set_xticks(range(9, 20), [f"{h:02d}:00" for h in range(9, 20)])
    ax.grid(axis="x", color=GRID)
    ax.legend(handles=[Patch(color=GOOD, label="on time"), Patch(color=SIGNAL, label="late"),
                       plt.Line2D([], [], marker="|", ls="", color=INK, ms=8, label="sighting (event)"),
                       plt.Line2D([], [], marker="|", ls="", color=MUTED, ms=5, label="duplicate (cooldown)"),
                       plt.Line2D([], [], marker="v", ls="", color=BAD, label="unknown"),
                       plt.Line2D([], [], marker="v", ls="", color=PURPLE, label="ambiguous (near twin)"),
                       plt.Line2D([], [], marker="v", ls="", color=MUTED, label="no face (real frame)")],
              frameon=False, fontsize=8, ncol=4, loc="lower center", bbox_to_anchor=(0.5, -0.17))
    ax.set_title(f"GET /v1/attendance/{demo['report']['date']}: check-in → check-out per employee "
                 "(synthetic day, times IST)")
    return _save(fig, out, "attendance_day.png")


def decisions_latency(demo: dict, out: Path) -> Path:
    s = summary(demo)
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12.5, 3.9), gridspec_kw={"width_ratios": [1.3, 1]})
    cats = [("check-in", s["events"].get("check_in", 0), GOOD), ("seen (later sighting)", s["events"].get("seen", 0),
                                                                   ACCENT),
            ("duplicate (cooldown)", s["events"].get("duplicate", 0), MUTED),
            ("unknown", s["decisions"].get("unknown", 0), BAD), ("ambiguous", s["decisions"].get("ambiguous", 0),
                                                               PURPLE),
            ("no face", s["decisions"].get("no_face", 0), SIGNAL)]
    a1.barh([c[0] for c in cats], [c[1] for c in cats], color=[c[2] for c in cats], height=0.6)
    for i, c in enumerate(cats):
        a1.text(c[1], i, f" {c[1]}", va="center", fontsize=9)
    a1.invert_yaxis()
    a1.grid(axis="x", color=GRID)
    a1.set_title(f"{s['sightings']} sightings → decisions (0 wrong-person accepts)")
    p = demo["plumbing"]
    stages = [("YuNet detect\n(640×480)", p["detect_ms_median_640x480"]), ("SFace embed\n(ONNX Runtime)",
                                                                           p["embed_ms_median"])]
    a2.bar([x[0] for x in stages], [x[1] for x in stages], color=[PURPLE, ACCENT], width=0.55)
    for i, (_, v) in enumerate(stages):
        a2.text(i, v, f"{v:.1f} ms", ha="center", va="bottom", fontsize=10)
    a2.set_ylabel("median ms (CPU, this build machine)")
    a2.grid(axis="y", color=GRID)
    a2.set_title("Real model latency per stage")
    return _save(fig, out, "decisions_latency.png")


def render_all(demo_json: Path, out: Path) -> list[Path]:
    demo = json.loads(Path(demo_json).read_text())
    out.mkdir(parents=True, exist_ok=True)
    return [architecture(out), similarity(demo, out), timeline(demo, out), decisions_latency(demo, out)]
