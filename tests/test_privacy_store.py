from datetime import datetime, timezone

import numpy as np

from attendance.store import Store
from tests.conftest import A, D, emb


def test_purge_retention(client, ids):
    client.post("/v1/recognize", json={"embedding": emb(ids.stranger()), "ts": "2026-01-01T05:00:00+00:00"},
                headers=D)
    client.post("/v1/recognize", json={"embedding": emb(ids.stranger()), "ts": "2026-11-05T05:00:00+00:00"},
                headers=D)
    client.post("/v1/recognize", json={"embedding": emb(ids.stranger()), "ts": "2026-11-16T05:00:00+00:00"},
                headers=D)
    svc = client.app.state.service
    out = svc.purge("admin", now=datetime(2026, 11, 16, 12, tzinfo=timezone.utc))
    assert out == {"events_deleted": 1, "unmatched_deleted": 1}             # 90-day and 7-day rules
    assert len(svc.store.q("SELECT * FROM events")) == 1


def test_no_images_are_stored(client, ids):
    import cv2

    pid = client.post("/v1/people", json={"external_id": "x", "name": "x"}, headers=A).json()["id"]
    client.post(f"/v1/people/{pid}/templates", json={"embedding": emb(ids.capture(1, 0.5))}, headers=A)
    store = client.app.state.service.store
    blobs = store.q("SELECT embedding FROM templates")
    assert all(len(b["embedding"]) == 128 * 4 for b in blobs)              # 512 bytes of float32, nothing else
    tables = {r["name"] for r in store.q("SELECT name FROM sqlite_master WHERE type = 'table'")}
    assert tables == {"people", "templates", "events", "audit"}
    del cv2


def test_templates_round_trip(tmp_path):
    s = Store(tmp_path / "s.db")
    pid = s.add_person("a", "A")
    v = np.arange(128, dtype=np.float32)
    s.add_template(pid, v, "embedding", None)
    t, p = s.templates()
    assert np.array_equal(t[0], v) and p.tolist() == [pid]
