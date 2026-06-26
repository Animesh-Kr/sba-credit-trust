"""Tests for rigorous (E-1) and adaptive (E-9) conformal."""
import numpy as np

from src.conformal.adaptive import AdaptiveConformalSelector, _CalibrationMap
from src.conformal.selective import SelectiveRiskController


def test_calibration_map_threshold_monotone_in_budget():
    rng = np.random.default_rng(0)
    p = rng.uniform(size=5000)
    y = (rng.uniform(size=5000) < p).astype(int)  # calibrated
    cmap = _CalibrationMap(p, y)
    # a larger risk budget allows a higher (more permissive) threshold
    assert cmap.threshold_for(0.30) >= cmap.threshold_for(0.05)


def test_rigorous_threshold_is_more_conservative_than_heuristic():
    rng = np.random.default_rng(1)
    p = rng.beta(2, 6, size=20000)
    y = (rng.uniform(size=20000) < p).astype(int)
    heur = SelectiveRiskController(alpha=0.05, delta=0.05).fit(p, y)
    rig = SelectiveRiskController(alpha=0.05, delta=0.05).fit_rigorous(p, y)
    # union bound -> never more permissive than the heuristic scan
    assert rig.t_lo <= heur.t_lo + 1e-9


def test_aci_controls_tail_risk_and_beats_static_under_shift():
    from src.conformal.adaptive import _CalibrationMap

    rng = np.random.default_rng(2)
    pc = rng.beta(2, 8, size=10000)            # calibration era: model is calibrated
    yc = (rng.uniform(size=10000) < pc).astype(int)
    sel = AdaptiveConformalSelector(pc, yc, alpha_target=0.05, gamma=0.2)
    static_map = _CalibrationMap(pc, yc)
    t_static = static_map.threshold_for(0.05)

    # future: the model under-predicts risk (miscalibrated under shift) — true default ~2.5x the
    # model probability. Static threshold (trusting the model) lets in loans that default more.
    aci_rates, static_rates = [], []
    for _ in range(60):
        pf = rng.beta(2, 8, size=1000)         # same model-output distribution
        p_true = np.clip(2.5 * pf, 0, 1)
        yf = (rng.uniform(size=1000) < p_true).astype(int)
        aci_rates.append(sel.step(pf, yf)["default_rate_among_approved"])
        appr = pf <= t_static
        static_rates.append(float(yf[appr].mean()) if appr.any() else 0.0)

    tail_aci = float(np.mean(aci_rates[-20:]))
    tail_static = float(np.mean(static_rates[-20:]))
    assert tail_aci <= 0.08            # ACI tail near target
    assert tail_aci < tail_static      # and clearly better than the static threshold under shift
