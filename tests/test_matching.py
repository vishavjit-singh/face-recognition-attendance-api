import numpy as np
import pytest

from attendance.config import Matching
from attendance.matching import Gallery, normalise

M = Matching(threshold=0.363, margin=0.05, top_k=3, max_templates_per_person=10)


def unit(*xs):
    v = np.zeros(128, np.float32)
    v[: len(xs)] = xs
    return v / np.linalg.norm(v)


def test_normalise_rejects_zero_and_nan():
    with pytest.raises(ValueError):
        normalise(np.zeros(128))
    with pytest.raises(ValueError):
        normalise(np.full(128, np.nan))
    assert np.isclose(np.linalg.norm(normalise(np.ones(128))), 1)


def test_best_template_per_person_and_ranking():
    g = Gallery(np.stack([unit(1, 0), unit(0.6, 0.8), unit(0, 1)]), np.array([1, 1, 2]))
    ranked = g.scores(unit(0.5, 0.86))
    assert [p for p, _ in ranked] == [1, 2]
    assert ranked[0][1] == pytest.approx(float(unit(0.6, 0.8) @ unit(0.5, 0.86)), abs=1e-6)


def test_accept_unknown_and_ambiguous():
    g = Gallery(np.stack([unit(1, 0, 0), unit(0, 1, 0)]), np.array([1, 2]))
    assert g.identify(unit(1, 0.1, 0), M).decision == "accepted"
    assert g.identify(unit(1, 0.1, 0), M).person_id == 1
    assert g.identify(unit(0, 0, 1), M).decision == "unknown"
    amb = g.identify(unit(1, 1.02, 0), M)          # both ≈ 0.71, gap < margin
    assert amb.decision == "ambiguous" and amb.person_id is None


def test_runner_up_below_threshold_does_not_make_it_ambiguous():
    g = Gallery(np.stack([unit(1, 0, 0), unit(0, 1, 0)]), np.array([1, 2]))
    m = g.identify(unit(1, 0.3, 0), M)
    assert m.decision == "accepted" and m.runner_up < M.threshold


def test_empty_gallery():
    m = Gallery(np.zeros((0, 128), np.float32), np.zeros(0, np.int64)).identify(unit(1), M)
    assert m.decision == "unknown"
