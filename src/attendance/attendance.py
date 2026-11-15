"""Turning accepted matches into check-ins and check-outs, and daily reports."""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

from .config import Attendance
from .store import Store


def parse_ts(ts: str | None) -> datetime:
    if not ts:
        return datetime.now(timezone.utc)
    t = datetime.fromisoformat(ts.replace("Z", "+00:00"))
    return t if t.tzinfo else t.replace(tzinfo=timezone.utc)


def local_date(t: datetime, a: Attendance) -> str:
    return (t + timedelta(minutes=a.timezone_offset_minutes)).date().isoformat()


def classify(store: Store, person_id: int, t: datetime, a: Attendance) -> str:
    """check_in | seen | duplicate for an accepted sighting at time t.

    The first sighting of the day is the check-in. Later sightings are kept as ``seen``; the report
    derives the check-out as the last one at least ``min_session_minutes`` after check-in (first in,
    last out), so a mid-day walk past the camera never ends someone's day early.
    """
    day = local_date(t, a)
    rows = store.q("SELECT ts, kind FROM events WHERE person_id = ? AND local_date = ? AND decision = 'accepted' "
                   "ORDER BY ts", (person_id, day))
    if rows:
        last = parse_ts(rows[-1]["ts"])
        if abs((t - last).total_seconds()) < a.cooldown_seconds:
            return "duplicate"
    if not any(r["kind"] == "check_in" for r in rows):
        return "check_in"
    return "seen"


def daily_report(store: Store, day: str, a: Attendance) -> dict:
    off = timedelta(minutes=a.timezone_offset_minutes)
    hh, mm = map(int, a.workday_start.split(":"))
    late_after = timedelta(hours=hh, minutes=mm + a.late_after_minutes)
    people = store.people()
    rows = []
    for p in people:
        ev = store.q("SELECT ts, kind FROM events WHERE person_id = ? AND local_date = ? AND decision = 'accepted' "
                     "AND kind IN ('check_in', 'seen') ORDER BY ts", (p["id"], day))
        cin = next((parse_ts(e["ts"]) for e in ev if e["kind"] == "check_in"), None)
        later = [parse_ts(e["ts"]) for e in ev if e["kind"] == "seen" and cin
                 and parse_ts(e["ts"]) - cin >= timedelta(minutes=a.min_session_minutes)]
        cout = later[-1] if later else None
        if cin is None:
            status = "absent"
        else:
            local_in = cin + off
            since_midnight = local_in - local_in.replace(hour=0, minute=0, second=0, microsecond=0)
            status = "late" if since_midnight > late_after else "on_time"
        rows.append({
            "person_id": p["id"], "external_id": p["external_id"], "name": p["name"], "status": status,
            "check_in": (cin + off).strftime("%H:%M") if cin else None,
            "check_out": (cout + off).strftime("%H:%M") if cout else None,
            "hours": round((cout - cin).total_seconds() / 3600, 2) if cin and cout else None,
            "missing_check_out": bool(cin and not cout),
        })
    counts = {s: sum(1 for r in rows if r["status"] == s) for s in ("on_time", "late", "absent")}
    decisions = {r["decision"]: r["n"] for r in store.q(
        "SELECT decision, COUNT(*) AS n FROM events WHERE local_date = ? GROUP BY decision", (day,))}
    dupes = store.q("SELECT COUNT(*) AS n FROM events WHERE local_date = ? AND kind = 'duplicate'", (day,))[0]["n"]
    return {"date": day, "people": rows, "summary": {**counts, "enrolled": len(people), "decisions": decisions,
                                                     "duplicates_suppressed": dupes}}
