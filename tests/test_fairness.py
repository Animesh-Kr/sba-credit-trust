"""Tests for the fairness disparity metrics (E-10)."""
import numpy as np

from src.eval import fairness


def test_group_metrics_and_zero_disparity_when_identical():
    rng = np.random.default_rng(0)
    n = 6000
    y = (rng.uniform(size=n) < 0.2).astype(int)
    p = np.clip(0.2 + 0.5 * y + rng.normal(0, 0.1, n), 0, 1)
    g = np.where(np.arange(n) % 2 == 0, "A", "B")  # groups independent of y, p
    rows = fairness.group_metrics(y, p, g, threshold=0.5, min_n=100)
    assert set(rows) == {"A", "B"}
    disp = fairness.disparity(rows)
    # identical distributions -> small gaps
    assert disp["approval_rate_gap"] < 0.05
    assert disp["ece_gap"] < 0.05


def test_disparity_detects_unfair_group():
    rng = np.random.default_rng(1)
    n = 6000
    y = (rng.uniform(size=n) < 0.2).astype(int)
    p = np.clip(0.2 + 0.5 * y + rng.normal(0, 0.1, n), 0, 1)
    g = np.array(["A"] * n, dtype=object)
    # group B systematically scored higher risk (lower approval) regardless of truth
    mask_b = np.arange(n) % 2 == 0
    g[mask_b] = "B"
    p[mask_b] = np.clip(p[mask_b] + 0.4, 0, 1)
    rows = fairness.group_metrics(y, p, g, threshold=0.5, min_n=100)
    disp = fairness.disparity(rows)
    assert disp["approval_rate_gap"] > 0.1  # clear demographic-parity gap
