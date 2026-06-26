"""Tests for the trustworthy-ML layer: calibration, uncertainty/OOD, conformal selection."""
import numpy as np

from src.calibration.scaling import IsotonicCalibrator, TemperatureScaler, logit
from src.conformal.selective import (
    SelectiveRiskController,
    clopper_pearson_lower,
    clopper_pearson_upper,
)
from src.eval.metrics import expected_calibration_error
from src.models.uncertainty import MahalanobisOOD, predictive_entropy


# --- calibration ---
def test_temperature_scaling_reduces_ece_on_overconfident_logits():
    rng = np.random.default_rng(0)
    n = 20000
    y = (rng.uniform(size=n) < 0.3).astype(int)
    # well-ordered but over-confident logits (scaled up) -> miscalibrated
    base = np.where(y == 1, rng.normal(1.0, 1.0, n), rng.normal(-1.0, 1.0, n))
    logits = base * 4.0
    p_raw = 1 / (1 + np.exp(-logits))
    ts = TemperatureScaler().fit(logits, y)
    p_cal = ts.transform(logits)
    assert ts.T_ > 1.0  # should shrink over-confident logits
    assert expected_calibration_error(y, p_cal) < expected_calibration_error(y, p_raw)


def test_isotonic_is_monotone_and_logit_roundtrip():
    rng = np.random.default_rng(1)
    p = rng.uniform(size=5000)
    y = (rng.uniform(size=5000) < p).astype(int)
    iso = IsotonicCalibrator().fit(p, y)
    grid = np.linspace(0, 1, 200)
    out = iso.transform(grid)
    assert np.all(np.diff(out) >= -1e-9)  # non-decreasing
    # logit/sigmoid roundtrip
    assert np.allclose(1 / (1 + np.exp(-logit(p))), p, atol=1e-5)


# --- uncertainty / OOD ---
def test_predictive_entropy_peaks_at_half():
    assert predictive_entropy(np.array([0.5]))[0] > predictive_entropy(np.array([0.01]))[0]
    assert abs(predictive_entropy(np.array([0.5]))[0] - np.log(2)) < 1e-9


def test_mahalanobis_separates_in_and_out_of_distribution():
    rng = np.random.default_rng(2)
    feats = rng.normal(0, 1, size=(2000, 8))
    labels = (rng.uniform(size=2000) < 0.5).astype(int)
    ood = MahalanobisOOD().fit(feats, labels)
    in_score = ood.score(rng.normal(0, 1, size=(500, 8)))
    out_score = ood.score(rng.normal(8, 1, size=(500, 8)))  # far away -> OOD
    assert out_score.mean() > in_score.mean() * 3


# --- conformal selective prediction ---
def test_clopper_pearson_bounds_ordering():
    up = clopper_pearson_upper(5, 100, 0.05)
    lo = clopper_pearson_lower(5, 100, 0.05)
    assert 0 <= lo <= 0.05 <= up <= 1


def test_selective_controller_bounds_approved_default_rate():
    rng = np.random.default_rng(3)
    n = 40000
    # score correlated with truth; approving low scores should be safe
    y = (rng.uniform(size=n) < 0.2).astype(int)
    prob = np.clip(0.15 + 0.6 * y + rng.normal(0, 0.2, n), 0, 1)
    cut = n // 2
    ctrl = SelectiveRiskController(alpha=0.05, delta=0.05).fit(prob[:cut], y[:cut])
    res = ctrl.evaluate(prob[cut:], y[cut:])
    # the guarantee: realised default rate among approved <= alpha (small finite-sample slack)
    assert res["n_approved"] > 0
    assert res["default_rate_among_approved"] <= ctrl.alpha + 0.02
    assert 0.0 <= res["coverage"] <= 1.0
