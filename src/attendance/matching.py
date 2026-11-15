"""Open-set identification over an in-memory gallery of L2-normalised templates."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .config import Matching


@dataclass
class Match:
    decision: str                    # accepted | unknown | ambiguous
    person_id: int | None
    score: float
    runner_up: float | None
    candidates: list[tuple[int, float]]


def normalise(v) -> np.ndarray:
    v = np.asarray(v, np.float32).ravel()
    n = float(np.linalg.norm(v))
    if n == 0 or not np.isfinite(n):
        raise ValueError("embedding has zero or invalid norm")
    return v / n


class Gallery:
    def __init__(self, templates: np.ndarray, person_ids: np.ndarray):
        self.templates = templates.astype(np.float32)
        self.person_ids = person_ids

    def __len__(self) -> int:
        return len(set(self.person_ids.tolist()))

    def scores(self, probe: np.ndarray) -> list[tuple[int, float]]:
        """Best template score per person, highest first."""
        if not len(self.templates):
            return []
        sims = self.templates @ normalise(probe)
        best: dict[int, float] = {}
        for pid, s in zip(self.person_ids.tolist(), sims.tolist()):
            if s > best.get(pid, -2.0):
                best[pid] = s
        return sorted(best.items(), key=lambda kv: kv[1], reverse=True)

    def identify(self, probe: np.ndarray, m: Matching) -> Match:
        ranked = self.scores(probe)
        top = ranked[: m.top_k]
        if not ranked:
            return Match("unknown", None, -1.0, None, [])
        pid, best = ranked[0]
        second = ranked[1][1] if len(ranked) > 1 else None
        if best < m.threshold:
            return Match("unknown", None, best, second, top)
        if second is not None and second >= m.threshold and best - second < m.margin:
            return Match("ambiguous", None, best, second, top)
        return Match("accepted", pid, best, second, top)
