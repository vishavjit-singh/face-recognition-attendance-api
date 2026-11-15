"""The evaluation code runs on a fake loader here; the real Olivetti run happens in CI."""
import numpy as np
import pytest

from attendance.config import load_settings
from attendance.evaluate import identification, verification
from attendance.synthetic import Identities
from tests.conftest import needs_models


def test_metrics_on_separable_embeddings():
    ids = Identities(40, seed=4)
    emb = np.stack([ids.capture(i, 0.6) for i in range(40) for _ in range(10)])
    labels = np.repeat(np.arange(40), 10)
    v = verification(emb, labels, 0.363)
    assert v["genuine_pairs"] == 1800 and v["impostor_pairs"] == 78000
    assert v["roc_auc"] > 0.99 and v["at_configured_threshold"]["far"] < 0.01
    i = identification(emb, labels, load_settings())
    assert i["known_probes"] == 210 and i["unknown_probes"] == 100
    assert i["rank1_closed_set"] > 0.95 and i["false_accept_rate_unknowns"] < 0.05


@needs_models
def test_run_with_a_fake_loader(tmp_path):
    pytest.importorskip("sklearn")
    from attendance.evaluate import run

    rng = np.random.default_rng(0)
    images = rng.random((40, 64, 64)).astype(np.float32)        # noise: nothing detected → fallback path
    labels = np.repeat(np.arange(4), 10)
    out = run(load_settings(), tmp_path, loader=lambda: (images, labels))
    assert out["detected"] == 0 and out["images"] == 40
    assert (tmp_path / "evaluation.json").exists() and (tmp_path / "evaluation_roc.png").exists()
