"""Tests for cost-sensitive dollar-aware decisioning."""
import numpy as np

from src.eval import cost


def test_realised_dollars_accounting():
    # 2 approved: one paid ($100k, margin 5% -> +5000), one defaulted (charged off $40k -> -40000)
    approve = np.array([True, True, False])
    y = np.array([0, 1, 1])
    gr = np.array([100_000.0, 80_000.0, 50_000.0])
    co = np.array([0.0, 40_000.0, 50_000.0])
    r = cost.realised_dollars(approve, y, gr, co, margin=0.05)
    assert r["profit_from_paid"] == 5000.0
    assert r["loss_from_defaults"] == 40000.0
    assert r["net_dollars"] == 5000.0 - 40000.0
    assert r["n_approved"] == 2
    assert r["n_defaults_approved"] == 1


def test_expected_value_decision_rejects_high_risk_high_exposure():
    # high p + high SBA exposure -> reject; low p + decent margin -> approve
    p = np.array([0.9, 0.02])
    gr = np.array([100_000.0, 100_000.0])
    sba = np.array([90_000.0, 90_000.0])
    appr = cost.expected_value_decision(p, gr, sba, margin=0.05)
    assert appr[0] == False  # noqa: E712  (expected loss dominates)
    assert appr[1] == True   # noqa: E712


def test_dollar_aware_beats_threshold_on_synthetic_portfolio():
    # When the decision-time LGD proxy (SBA_Appv) equals the realised loss, the EV rule is the
    # Bayes-optimal per-loan decision and so maximises realised $ among policies.
    rng = np.random.default_rng(0)
    n = 20000
    p = rng.beta(2, 6, size=n)               # true default probability per loan
    y = (rng.uniform(size=n) < p).astype(int)  # outcome drawn from p -> p is calibrated
    gr = rng.uniform(20_000, 500_000, n)
    sba = gr * rng.uniform(0.2, 0.5, n)     # guaranteed portion
    co = np.where(y == 1, sba, 0.0)          # realised loss given default = guaranteed amount
    res = cost.compare_policies(p, y, gr, sba, co, margin=0.15)
    ev = res["dollar_aware_EV"]["net_dollars"]
    thr = res["threshold_0.5"]["net_dollars"]
    allp = res["approve_all"]["net_dollars"]
    assert ev >= thr        # EV policy at least matches the accuracy threshold in $
    assert ev >= allp       # and beats approving everyone
    # and it approves fewer defaulters than blanket approval
    assert res["dollar_aware_EV"]["n_defaults_approved"] < res["approve_all"]["n_defaults_approved"]
