"""Settings from config/settings.yaml, with API keys from the environment."""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = ROOT / "config" / "settings.yaml"


@dataclass(frozen=True)
class ModelFile:
    file: str
    url: str
    sha256: str


@dataclass(frozen=True)
class Quality:
    min_face_px: int
    min_sharpness: float
    brightness: tuple[float, float]
    max_yaw: float
    max_roll_deg: float


@dataclass(frozen=True)
class Matching:
    threshold: float
    margin: float
    top_k: int
    max_templates_per_person: int


@dataclass(frozen=True)
class Attendance:
    timezone_offset_minutes: int
    cooldown_seconds: int
    min_session_minutes: int
    workday_start: str
    late_after_minutes: int


@dataclass(frozen=True)
class Privacy:
    store_images: bool
    event_retention_days: int
    unknown_retention_days: int


@dataclass(frozen=True)
class Settings:
    model_dir: Path
    detector: ModelFile
    recognizer: ModelFile
    score_threshold: float
    nms_threshold: float
    max_side: int
    quality: Quality
    matching: Matching
    attendance: Attendance
    privacy: Privacy
    accept_embeddings: bool = True
    max_image_bytes: int = 5_000_000
    database: Path = ROOT / "data" / "attendance.db"
    api_keys: dict[str, str] = field(default_factory=dict, repr=False)   # key -> role

    def model_path(self, which: str) -> Path:
        return self.model_dir / getattr(self, which).file


def _keys(env: dict) -> dict[str, str]:
    keys = {}
    for role, var in (("admin", "ATTENDANCE_ADMIN_KEY"), ("device", "ATTENDANCE_DEVICE_KEY")):
        for k in filter(None, (env.get(var) or "").split(",")):
            keys[k.strip()] = role
    return keys


def load_settings(path: str | Path | None = None, env: dict | None = None, **overrides) -> Settings:
    raw = yaml.safe_load(Path(path or DEFAULT_CONFIG).read_text())
    env = dict(os.environ if env is None else env)
    m, q, a, p, d = raw["models"], raw["quality"], raw["attendance"], raw["privacy"], raw["detection"]
    model_dir = Path(env.get("ATTENDANCE_MODEL_DIR") or m["dir"])
    s = Settings(
        model_dir=model_dir if model_dir.is_absolute() else ROOT / model_dir,
        detector=ModelFile(**m["detector"]), recognizer=ModelFile(**m["recognizer"]),
        score_threshold=float(d["score_threshold"]), nms_threshold=float(d["nms_threshold"]),
        max_side=int(d["max_side"]),
        quality=Quality(min_face_px=int(q["min_face_px"]), min_sharpness=float(q["min_sharpness"]),
                        brightness=(float(q["brightness"][0]), float(q["brightness"][1])),
                        max_yaw=float(q["max_yaw"]), max_roll_deg=float(q["max_roll_deg"])),
        matching=Matching(**raw["matching"]),
        attendance=Attendance(**a),
        privacy=Privacy(**p),
        accept_embeddings=bool(raw["api"]["accept_embeddings"]),
        max_image_bytes=int(raw["api"]["max_image_bytes"]),
        database=Path(env.get("ATTENDANCE_DB") or ROOT / raw["api"]["database"]),
        api_keys=_keys(env),
    )
    if overrides:
        s = Settings(**{**s.__dict__, **overrides})
    return s
