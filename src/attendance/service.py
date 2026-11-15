"""Application logic shared by the API, the CLI and the tests (no HTTP in here)."""
from __future__ import annotations

import threading
from datetime import datetime, timedelta, timezone

import numpy as np

from . import attendance
from .config import Settings
from .matching import Gallery, normalise
from .store import Store


class ServiceError(Exception):
    def __init__(self, status: int, code: str, message: str, **detail):
        super().__init__(message)
        self.status, self.code, self.message, self.detail = status, code, message, detail


class AttendanceService:
    def __init__(self, settings: Settings, store: Store, engine=None):
        self.settings, self.store = settings, store
        self._engine = engine
        self._engine_lock = threading.Lock()
        self._gallery: Gallery | None = None

    # ------------------------------------------------------------------ helpers
    @property
    def engine(self):
        with self._engine_lock:
            if self._engine is None:
                from .vision import FaceEngine

                self._engine = FaceEngine(self.settings)
            return self._engine

    @property
    def gallery(self) -> Gallery:
        if self._gallery is None:
            self._gallery = Gallery(*self.store.templates())
        return self._gallery

    def _invalidate(self) -> None:
        self._gallery = None

    def _embedding_from(self, image: bytes | None, embedding, require_single: bool):
        if embedding is not None:
            if not self.settings.accept_embeddings:
                raise ServiceError(403, "embeddings_disabled", "this server only accepts images")
            vec = np.asarray(embedding, dtype=np.float32).ravel()
            if vec.shape != (128,):
                raise ServiceError(422, "bad_embedding", f"expected 128 values, got {vec.size}")
            try:
                return normalise(vec), None, {}, "embedding"
            except ValueError as exc:
                raise ServiceError(422, "bad_embedding", str(exc)) from None
        if image is None:
            raise ServiceError(422, "missing_input", "send an image or an embedding")
        if len(image) > self.settings.max_image_bytes:
            raise ServiceError(413, "image_too_large", "image exceeds the configured size limit")
        a = self.engine.analyze(image, require_single=require_single)
        return a.embedding, a, a.timings_ms, a.status

    # ------------------------------------------------------------------ people
    def create_person(self, external_id: str, name: str, role: str) -> dict:
        if not external_id or not name:
            raise ServiceError(422, "validation_error", "external_id and name are required")
        if self.store.q("SELECT id FROM people WHERE external_id = ?", (external_id,)):
            raise ServiceError(409, "duplicate_person", "external_id already enrolled")
        pid = self.store.add_person(external_id, name)
        self.store.audit(role, "create_person", str(pid), {"external_id": external_id})
        return self.store.person(pid)

    def enroll(self, person_id: int, role: str, image: bytes | None = None, embedding=None,
               force: bool = False) -> dict:
        person = self.store.person(person_id)
        if not person:
            raise ServiceError(404, "not_found", "person not found")
        if person["templates"] >= self.settings.matching.max_templates_per_person:
            raise ServiceError(409, "too_many_templates", "template limit reached; delete and re-enrol")
        vec, analysis, timings, source = self._embedding_from(image, embedding, require_single=True)
        if vec is None:
            q = analysis.quality.as_dict() if analysis and analysis.quality else None
            raise ServiceError(422, analysis.status, "enrolment image rejected", faces=analysis.faces, quality=q)
        # a template that already matches someone else strongly is probably the wrong person
        others = [(pid, s) for pid, s in self.gallery.scores(vec) if pid != person_id]
        clash = others and others[0][1] >= self.settings.matching.threshold + self.settings.matching.margin
        if clash and not force:
            raise ServiceError(409, "matches_other_person", "this face already matches another enrolled person",
                               other_person_id=others[0][0], score=round(others[0][1], 4))
        q = analysis.quality.as_dict() if analysis and analysis.quality else None
        tid = self.store.add_template(person_id, vec, "image" if analysis else "embedding", q)
        self._invalidate()
        detail = {"template_id": tid, "source": source}
        if clash:
            detail.update(forced_despite_match=others[0][0], score=round(others[0][1], 4))
        self.store.audit(role, "enroll", str(person_id), detail)
        return {"person_id": person_id, "template_id": tid, "templates": person["templates"] + 1, "quality": q,
                "timings_ms": {k: round(v, 2) for k, v in timings.items()}}

    def delete_person(self, person_id: int, role: str) -> dict:
        if not self.store.person(person_id):
            raise ServiceError(404, "not_found", "person not found")
        counts = self.store.delete_person(person_id)
        self._invalidate()
        self.store.audit(role, "delete_person", str(person_id), counts)
        return {"deleted": person_id, **counts}

    # ------------------------------------------------------------------ recognition
    def recognize(self, device_id: str, image: bytes | None = None, embedding=None, ts: str | None = None) -> dict:
        t = attendance.parse_ts(ts)
        a = self.settings.attendance
        day = attendance.local_date(t, a)
        vec, analysis, timings, status = self._embedding_from(image, embedding, require_single=False)
        base = {"ts": t.isoformat(timespec="seconds"), "local_date": day, "device_id": device_id}
        if vec is None:
            detail = ",".join(analysis.quality.reasons) if analysis and analysis.quality else None
            self.store.add_event(**base, decision=status, detail=detail)
            return {"decision": status, "faces": analysis.faces if analysis else 0, "detail": detail,
                    "timings_ms": {k: round(v, 2) for k, v in timings.items()}}
        m = self.gallery.identify(vec, self.settings.matching)
        out = {"decision": m.decision, "score": round(m.score, 4),
               "runner_up": None if m.runner_up is None else round(m.runner_up, 4),
               "timings_ms": {k: round(v, 2) for k, v in timings.items()}}
        if m.decision != "accepted":
            self.store.add_event(**base, decision=m.decision, score=m.score, runner_up=m.runner_up)
            return out
        kind = attendance.classify(self.store, m.person_id, t, a)
        self.store.add_event(**base, decision="accepted", kind=kind, person_id=m.person_id, score=m.score,
                             runner_up=m.runner_up)
        person = self.store.person(m.person_id)
        return {**out, "person_id": m.person_id, "external_id": person["external_id"], "name": person["name"],
                "event": kind}

    # ------------------------------------------------------------------ reports and housekeeping
    def report(self, day: str) -> dict:
        return attendance.daily_report(self.store, day, self.settings.attendance)

    def purge(self, role: str, now: datetime | None = None) -> dict:
        now = now or datetime.now(timezone.utc)
        p = self.settings.privacy
        out = self.store.purge((now - timedelta(days=p.event_retention_days)).isoformat(timespec="seconds"),
                               (now - timedelta(days=p.unknown_retention_days)).isoformat(timespec="seconds"))
        self.store.audit(role, "purge", None, out)
        return out
