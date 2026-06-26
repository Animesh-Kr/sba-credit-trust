"""Tests for bootstrap CIs and seed summaries (E-2)."""
import numpy as np
from sklearn.metrics import roc_auc_score

from src.eval.stats import bootstrap_ci, seed_summary


def test_bootstrap_ci_contains_point_and_orders():
    rng = np.random.default_rng(0)
    y = (rng.uniform(size=4000) < 0.3).astype(int)
    p = np.clip(0.3 + 0.5 * y + rng.normal(0, 0.2, 4000), 0, 1)
    pt, lo, hi = bootstrap_ci(y, p, roc_auc_score, n_boot=300, seed=1)
    assert lo <= pt <= hi
    assert 0.0 <= lo < hi <= 1.0


def test_bootstrap_ci_narrows_with_more_data():
    rng = np.random.default_rng(1)

    def width(n):
        y = (rng.uniform(size=n) < 0.3).astype(int)
        p = np.clip(0.3 + 0.5 * y + rng.normal(0, 0.2, n), 0, 1)
        _, lo, hi = bootstrap_ci(y, p, roc_auc_score, n_boot=300, seed=2)
        return hi - lo

    assert width(8000) < width(1000)


def test_seed_summary_stats():
    dicts = [{"a": 1.0}, {"a": 2.0}, {"a": 3.0}]
    s = seed_summary(dicts)["a"]
    assert s["mean"] == 2.0
    assert s["min"] == 1.0 and s["max"] == 3.0
    assert s["std"] > 0
