"""Download the pinned ONNX models and verify their SHA-256 before anything loads them."""
from __future__ import annotations

import hashlib
import urllib.request
from pathlib import Path

from .config import ModelFile, Settings


class ModelIntegrityError(RuntimeError):
    pass


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def verify(path: Path, spec: ModelFile) -> None:
    if not path.exists():
        raise FileNotFoundError(f"{path} is missing: run `make models`")
    digest = sha256(path)
    if digest != spec.sha256:
        raise ModelIntegrityError(f"{path.name}: sha256 {digest} does not match the pinned {spec.sha256}")


def ensure(settings: Settings, opener=urllib.request.urlopen) -> list[Path]:
    """Download missing models (to a temp file, verified, then renamed). Existing files are re-verified."""
    settings.model_dir.mkdir(parents=True, exist_ok=True)
    out = []
    for spec in (settings.detector, settings.recognizer):
        path = settings.model_dir / spec.file
        if not path.exists():
            tmp = path.with_suffix(".part")
            with opener(spec.url) as resp, tmp.open("wb") as f:
                while chunk := resp.read(1 << 20):
                    f.write(chunk)
            if sha256(tmp) != spec.sha256:
                tmp.unlink()
                raise ModelIntegrityError(f"{spec.file}: downloaded file does not match the pinned sha256")
            tmp.rename(path)
        verify(path, spec)
        out.append(path)
    return out
