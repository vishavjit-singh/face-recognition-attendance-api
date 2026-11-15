from tests.conftest import A, D, emb


def enrol(client, ids, i, n=3, ext=None):
    r = client.post("/v1/people", json={"external_id": ext or f"E{i}", "name": f"Person {i}"}, headers=A)
    assert r.status_code == 201, r.text
    pid = r.json()["id"]
    for _ in range(n):
        r = client.post(f"/v1/people/{pid}/templates", json={"embedding": emb(ids.capture(i, 0.5)), "force": True},
                        headers=A)
        assert r.status_code == 201, r.text
    return pid


def see(client, vec, ts):
    r = client.post("/v1/recognize", json={"embedding": emb(vec), "device_id": "gate", "ts": ts}, headers=D)
    assert r.status_code == 200, r.text
    return r.json()


def test_auth_roles(client):
    assert client.get("/v1/people").status_code == 401
    assert client.get("/v1/people", headers={"X-API-Key": "nope"}).json()["error"]["code"] == "invalid_api_key"
    assert client.get("/v1/people", headers=D).status_code == 403            # devices can't list people
    assert client.post("/v1/recognize", json={"embedding": [1.0] * 128}, headers=D).status_code == 200
    assert client.get("/health").json()["status"] == "ok"


def test_validation(client, ids):
    assert client.post("/v1/people", json={"external_id": "", "name": "x"}, headers=A).status_code == 422
    enrol(client, ids, 2, n=1, ext="dup")
    assert client.post("/v1/people", json={"external_id": "dup", "name": "y"}, headers=A).status_code == 409
    r = client.post("/v1/recognize", json={"embedding": [1, 2, 3]}, headers=D)
    assert r.status_code == 422 and r.json()["error"]["code"] == "bad_embedding"
    assert client.post("/v1/recognize", json={"embedding": [0.0] * 128}, headers=D).status_code == 422
    assert client.post("/v1/recognize", content=b"x", headers={**D, "content-type": "text/plain"}).status_code == 415
    assert client.post("/v1/people/999/templates", json={"embedding": [1.0] * 128}, headers=A).status_code == 404
    assert client.get("/v1/attendance/16-11-2026", headers=A).status_code == 422


def test_recognition_and_attendance_day(client, ids):
    p3, p4 = enrol(client, ids, 3), enrol(client, ids, 4)
    enrol(client, ids, 5)                                                       # absent today
    day = "2026-11-16"
    first = see(client, ids.capture(3, 0.8), f"{day}T03:45:00+00:00")         # 09:15 IST
    assert first["decision"] == "accepted" and first["person_id"] == p3 and first["event"] == "check_in"
    assert see(client, ids.capture(3, 0.8), f"{day}T03:45:20+00:00")["event"] == "duplicate"
    assert see(client, ids.capture(4, 0.8), f"{day}T04:20:00+00:00")["event"] == "check_in"   # 09:50: late
    assert see(client, ids.capture(3, 0.8), f"{day}T07:30:00+00:00")["event"] == "seen"       # lunch
    assert see(client, ids.capture(3, 0.8), f"{day}T12:05:00+00:00")["event"] == "seen"       # 17:35 leaves
    assert see(client, ids.stranger(), f"{day}T05:00:00+00:00")["decision"] == "unknown"
    rep = client.get(f"/v1/attendance/{day}", headers=A).json()
    rows = {r["person_id"]: r for r in rep["people"]}
    assert rows[p3]["status"] == "on_time" and rows[p3]["check_in"] == "09:15" and rows[p3]["check_out"] == "17:35"
    assert rows[p3]["hours"] == 8.33
    assert rows[p4]["status"] == "late" and rows[p4]["missing_check_out"]
    assert rep["summary"]["absent"] == 1 and rep["summary"]["duplicates_suppressed"] == 1
    assert rep["summary"]["decisions"] == {"accepted": 5, "unknown": 1}


def test_check_out_needs_minimum_session(client, ids):
    p = enrol(client, ids, 6)
    see(client, ids.capture(6, 0.7), "2026-11-16T04:00:00+00:00")
    see(client, ids.capture(6, 0.7), "2026-11-16T04:10:00+00:00")          # 10 min later: not a check-out
    row = client.get("/v1/attendance/2026-11-16", headers=A).json()["people"][0]
    assert row["person_id"] == p and row["check_out"] is None


def test_near_twin_guard_and_forced_enrolment(client, ids):
    a = client.post("/v1/people", json={"external_id": "twin-a", "name": "A"}, headers=A).json()["id"]
    b = client.post("/v1/people", json={"external_id": "twin-b", "name": "B"}, headers=A).json()["id"]
    client.post(f"/v1/people/{a}/templates", json={"embedding": emb(ids.capture(0, 0.4))}, headers=A)
    r = client.post(f"/v1/people/{b}/templates", json={"embedding": emb(ids.capture(1, 0.4))}, headers=A)
    assert r.status_code == 409 and r.json()["error"]["code"] == "matches_other_person"
    r = client.post(f"/v1/people/{b}/templates", json={"embedding": emb(ids.capture(1, 0.4)), "force": True},
                    headers=A)
    assert r.status_code == 201
    audit = client.get("/v1/admin/audit", headers=A).json()["audit"]
    assert '"forced_despite_match"' in audit[0]["detail"]


def test_template_limit(client, ids, settings):
    pid = enrol(client, ids, 7, n=settings.matching.max_templates_per_person)
    r = client.post(f"/v1/people/{pid}/templates", json={"embedding": emb(ids.capture(7, 0.5))}, headers=A)
    assert r.status_code == 409 and r.json()["error"]["code"] == "too_many_templates"


def test_right_to_erasure(client, ids):
    pid = enrol(client, ids, 8)
    see(client, ids.capture(8, 0.7), "2026-11-16T04:00:00+00:00")
    r = client.request("DELETE", f"/v1/people/{pid}", headers=A).json()
    assert r == {"deleted": pid, "templates": 3, "events": 1}
    assert see(client, ids.capture(8, 0.7), "2026-11-16T05:00:00+00:00")["decision"] == "unknown"
    actions = [a["action"] for a in client.get("/v1/admin/audit", headers=A).json()["audit"]]
    assert actions[0] == "delete_person"
    assert client.request("DELETE", f"/v1/people/{pid}", headers=A).status_code == 404


def test_embeddings_can_be_disabled(settings, tmp_path):
    from starlette.testclient import TestClient

    from attendance.api import create_app
    from attendance.config import Settings
    from attendance.store import Store

    s = Settings(**{**settings.__dict__, "accept_embeddings": False})
    with TestClient(create_app(s, Store(tmp_path / "x.db"))) as c:
        r = c.post("/v1/recognize", json={"embedding": [1.0] * 128}, headers=D)
        assert r.status_code == 403 and r.json()["error"]["code"] == "embeddings_disabled"
