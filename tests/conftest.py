import numpy as np
import pytest
from starlette.testclient import TestClient

from attendance.api import create_app
from attendance.config import Settings, load_settings
from attendance.store import Store
from attendance.synthetic import Identities

ADMIN, DEVICE = "test-admin", "test-device"


def models_present() -> bool:
    s = load_settings()
    return s.model_path("detector").exists() and s.model_path("recognizer").exists()


needs_models = pytest.mark.skipif(not models_present(), reason="run `make models` first")


@pytest.fixture
def settings(tmp_path) -> Settings:
    s = load_settings(env={})
    return Settings(**{**s.__dict__, "api_keys": {ADMIN: "admin", DEVICE: "device"},
                       "database": tmp_path / "t.db"})


@pytest.fixture
def client(settings):
    with TestClient(create_app(settings, Store(settings.database))) as c:
        yield c


@pytest.fixture
def ids():
    return Identities(10, seed=1, lookalikes=((0, 1),))


A = {"X-API-Key": ADMIN}
D = {"X-API-Key": DEVICE}


def emb(v: np.ndarray) -> list:
    return [float(x) for x in v]
