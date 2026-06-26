"""Uncertainty on the metrics themselves: bootstrap CIs and across-seed summaries.

Phase-A–D numbers were point estimates from a single split/seed (AUDIT M-2). This module adds the
two standard forms of metric uncertainty:

* **Bootstrap CI** — resample the test set with replacement and recompute a metric, giving a
  confidence interval that captures finite-test-set variance (cheap; one trained model).
* **Across-seed summary** — mean ± std of a metric over independent train/split seeds, capturing
  training + split variance (expensive; needs many fits).
"""
from __future__ import annotations

import numpy as np


def bootstrap_ci(y_true, y_prob, metric_fn, n_boot: int = 1000, alpha: float = 0.05, seed: int = 0):
    """Percentile bootstrap CI for a metric of the form ``metric_fn(y_true, y_prob) -> float``.

    Returns (point_estimate, lo, hi) for the central ``1 - alpha`` interval.
    """
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob, dtype=np.float64)
    n = len(y_true)
    rng = np.random.default_rng(seed)
    point = float(metric_fn(y_true, y_prob))
    stats = np.empty(n_boot, dtype=np.float64)
    for b in range(n_boot):
        idx = rng.integers(0, n, n)
        # guard degenerate resamples (single class) for AUC-type metrics
        yt = y_true[idx]
        if yt.min() == yt.max():
            stats[b] = np.nan
            continue
        stats[b] = metric_fn(yt, y_prob[idx])
    lo, hi = np.nanpercentile(stats, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return point, float(lo), float(hi)


def seed_summary(metric_dicts: list[dict]) -> dict[str, dict]:
    """Mean/std/min/max for each metric key across a list of per-seed metric dicts."""
    keys = metric_dicts[0].keys()
    out = {}
    for k in keys:
        vals = np.array([d[k] for d in metric_dicts], dtype=np.float64)
        out[k] = {
            "mean": float(vals.mean()),
            "std": float(vals.std(ddof=1)) if len(vals) > 1 else 0.0,
            "min": float(vals.min()),
            "max": float(vals.max()),
        }
    return out
