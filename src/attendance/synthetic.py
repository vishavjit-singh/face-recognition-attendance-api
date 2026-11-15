"""SYNTHETIC embeddings for the demo and the tests: no faces, no people.

Each fictional identity is a random unit vector in the 128-d embedding space. A "capture" is that
centroid plus Gaussian noise, re-normalised. The noise level stands in for capture quality: with
noise σ the expected cosine between two captures of the same identity is about 1 / (1 + σ²).
This exercises the matching, margin, cooldown and reporting logic; it says nothing about how well
SFace separates real faces (that is what `evaluate` measures, in CI).
"""
from __future__ import annotations

import numpy as np

DIM = 128


class Identities:
    def __init__(self, n: int, seed: int = 9, lookalikes: tuple[tuple[int, int], ...] = ()):
        self.rng = np.random.default_rng(seed)
        c = self.rng.normal(size=(n, DIM))
        for a, b in lookalikes:                        # b becomes a near-twin of a
            c[b] = c[a] / np.linalg.norm(c[a]) + 0.5 * self.rng.normal(size=DIM) / np.sqrt(DIM)
        self.centroids = c / np.linalg.norm(c, axis=1, keepdims=True)

    def capture(self, i: int, sigma: float) -> np.ndarray:
        v = self.centroids[i] + sigma * self.rng.normal(size=DIM) / np.sqrt(DIM)
        return (v / np.linalg.norm(v)).astype(np.float32)

    def stranger(self) -> np.ndarray:
        v = self.rng.normal(size=DIM)
        return (v / np.linalg.norm(v)).astype(np.float32)
