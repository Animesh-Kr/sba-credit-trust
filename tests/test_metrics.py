"""Tests for the evaluation metrics — especially ECE, the project's hallmark."""
import numpy as np

from src.eval import metrics


def test_ece_perfectly_calibrated_is_low():
    # probs equal to empirical frequency -> ECE near 0
    rng = np.random.default_rng(0)
    p = rng.uniform(0, 1, size=20000)
    y = (rng.uniform(0, 1, size=20000) < p).astype(int)
    assert metrics.expected_calibration_error(y, p, n_bins=15) < 0.03


def test_ece_miscalibrated_is_high():
    # model always says 0.9 but truth is ~0.1 -> large calibration error
    y = np.zeros(1000, dtype=int)
    y[:100] = 1
    p = np.full(1000, 0.9)
    assert metrics.expected_calibration_error(y, p) > 0.5


def test_classification_report_keys_and_ranges():
    rng = np.random.default_rng(1)
    y = (rng.uniform(size=2000) < 0.2).astype(int)
    p = np.clip(y * 0.6 + rng.uniform(size=2000) * 0.4, 0, 1)
    rep = metrics.classification_report(y, p)
    for k in ["pr_auc", "roc_auc", "f1_minority", "brier", "ece", "prevalence"]:
        assert k in rep
    assert 0.0 <= rep["pr_auc"] <= 1.0
    assert 0.0 <= rep["roc_auc"] <= 1.0
    assert abs(rep["prevalence"] - y.mean()) < 1e-9


def test_perfect_ranker_has_top_scores():
    y = np.array([0, 0, 1, 1])
    p = np.array([0.1, 0.2, 0.8, 0.9])
    rep = metrics.classification_report(y, p)
    assert rep["roc_auc"] == 1.0
    assert rep["pr_auc"] == 1.0
