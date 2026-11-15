"""SQLite storage. Only embeddings are persisted (no images), and every admin action is audited."""
from __future__ import annotations

import json
import sqlite3
import threading
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

SCHEMA = """
PRAGMA foreign_keys = ON;
CREATE TABLE IF NOT EXISTS people (
    id          INTEGER PRIMARY KEY,
    external_id TEXT NOT NULL UNIQUE,
    name        TEXT NOT NULL,
    created_at  TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS templates (
    id         INTEGER PRIMARY KEY,
    person_id  INTEGER NOT NULL REFERENCES people(id) ON DELETE CASCADE,
    embedding  BLOB NOT NULL,
    source     TEXT NOT NULL,
    quality    TEXT,
    created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS events (
    id           INTEGER PRIMARY KEY,
    ts           TEXT NOT NULL,
    local_date   TEXT NOT NULL,
    device_id    TEXT NOT NULL,
    decision     TEXT NOT NULL,
    kind         TEXT,
    person_id    INTEGER REFERENCES people(id) ON DELETE CASCADE,
    score        REAL,
    runner_up    REAL,
    detail       TEXT
);
CREATE INDEX IF NOT EXISTS events_person_day ON events(person_id, local_date, ts);
CREATE TABLE IF NOT EXISTS audit (
    id     INTEGER PRIMARY KEY,
    ts     TEXT NOT NULL,
    role   TEXT NOT NULL,
    action TEXT NOT NULL,
    target TEXT,
    detail TEXT
);
"""


def utcnow() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class Store:
    def __init__(self, path: str | Path):
        if str(path) != ":memory:":
            Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(str(path), check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self.lock = threading.RLock()
        with self.lock:
            self.conn.executescript(SCHEMA)
            if str(path) != ":memory:":
                self.conn.execute("PRAGMA journal_mode=WAL")

    def q(self, sql: str, params=()) -> list[dict]:
        with self.lock:
            return [dict(r) for r in self.conn.execute(sql, params).fetchall()]

    def x(self, sql: str, params=()) -> int:
        with self.lock:
            cur = self.conn.execute(sql, params)
            self.conn.commit()
            return cur.lastrowid

    # ---------------------------------------------------------------- people and templates
    def add_person(self, external_id: str, name: str) -> int:
        return self.x("INSERT INTO people (external_id, name, created_at) VALUES (?, ?, ?)",
                      (external_id, name, utcnow()))

    def person(self, person_id: int) -> dict | None:
        rows = self.q("SELECT p.*, (SELECT COUNT(*) FROM templates t WHERE t.person_id = p.id) AS templates "
                      "FROM people p WHERE p.id = ?", (person_id,))
        return rows[0] if rows else None

    def people(self) -> list[dict]:
        return self.q("SELECT p.*, (SELECT COUNT(*) FROM templates t WHERE t.person_id = p.id) AS templates "
                      "FROM people p ORDER BY p.id")

    def add_template(self, person_id: int, embedding: np.ndarray, source: str, quality: dict | None) -> int:
        return self.x("INSERT INTO templates (person_id, embedding, source, quality, created_at) VALUES (?,?,?,?,?)",
                      (person_id, np.asarray(embedding, np.float32).tobytes(), source,
                       json.dumps(quality) if quality else None, utcnow()))

    def templates(self) -> tuple[np.ndarray, np.ndarray]:
        rows = self.q("SELECT person_id, embedding FROM templates ORDER BY id")
        if not rows:
            return np.zeros((0, 128), np.float32), np.zeros(0, np.int64)
        return (np.stack([np.frombuffer(r["embedding"], np.float32) for r in rows]),
                np.array([r["person_id"] for r in rows], np.int64))

    def delete_person(self, person_id: int) -> dict:
        """Right to erasure: the person, every template and every event about them are deleted."""
        with self.lock:
            counts = {"templates": self.conn.execute("SELECT COUNT(*) FROM templates WHERE person_id = ?",
                                                     (person_id,)).fetchone()[0],
                      "events": self.conn.execute("SELECT COUNT(*) FROM events WHERE person_id = ?",
                                                  (person_id,)).fetchone()[0]}
            self.conn.execute("DELETE FROM people WHERE id = ?", (person_id,))
            self.conn.commit()
        return counts

    # ---------------------------------------------------------------- events and audit
    def add_event(self, **e) -> int:
        cols = ["ts", "local_date", "device_id", "decision", "kind", "person_id", "score", "runner_up", "detail"]
        return self.x(f"INSERT INTO events ({', '.join(cols)}) VALUES ({', '.join('?' * len(cols))})",
                      tuple(e.get(c) for c in cols))

    def audit(self, role: str, action: str, target: str | None = None, detail: dict | None = None) -> None:
        self.x("INSERT INTO audit (ts, role, action, target, detail) VALUES (?, ?, ?, ?, ?)",
               (utcnow(), role, action, target, json.dumps(detail) if detail else None))

    def purge(self, before_all: str, before_unmatched: str) -> dict:
        with self.lock:
            a = self.conn.execute("DELETE FROM events WHERE ts < ?", (before_all,)).rowcount
            b = self.conn.execute("DELETE FROM events WHERE person_id IS NULL AND ts < ?", (before_unmatched,)).rowcount
            self.conn.commit()
        return {"events_deleted": a, "unmatched_deleted": b}
