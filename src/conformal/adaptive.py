"""E-9: Adaptive Conformal Inference for risk-controlled loan approval under distribution shift.

Phase D showed a *static* conformal threshold, calibrated on the pre-2008 era, lets the realised
default rate among auto-approved loans drift well above the target on later cohorts — split-conformal
assumes exchangeability, which temporal data violates.

Adaptive Conformal Inference (Gibbs & Candès, 2021) fixes this with online feedback: after each time
step it nudges a working risk budget up or down depending on whether the realised risk overshot the
target, with no exchangeability assumption. Applied here, the approval threshold tightens after a
high-default cohort and loosens after a safe one, keeping the *long-run* default rate among approved
loans near the target across the crisis.
"""
from __future__ import annotations

import numpy as np


class _CalibrationMap:
    """Maps a working risk budget a -> approval threshold t such that, on the calibration set, the
    empirical default rate among {p <= t} is ~a. Precomputed once for O(log n) lookups."""

    def __init__(self, prob_cal: np.ndarray, y_cal: np.ndarray):
        order = np.argsort(np.asarray(prob_cal, dtype=np.float64))
        self.p = np.asarray(prob_cal, dtype=np.float64)[order]
        y = np.asarray(y_cal, dtype=int)[order]
        n = np.arange(1, len(y) + 1)
        self.cum_rate = np.cumsum(y) / n  # default rate among the lowest-k probabilities

    def threshold_for(self, budget: float) -> float:
        """Largest threshold whose prefix default rate <= budget."""
        ok = np.where(self.cum_rate <= budget)[0]
        if len(ok) == 0:
            return 0.0
        return float(self.p[ok[-1]])


class AdaptiveConformalSelector:
    """Online approve-threshold controller (ACI) for non-exchangeable, time-ordered cohorts."""

    def __init__(self, prob_cal, y_cal, alpha_target: float = 0.05, gamma: float = 0.05):
        self.cmap = _CalibrationMap(prob_cal, y_cal)
        self.alpha_target = alpha_target
        self.gamma = gamma
        self.budget = alpha_target  # working risk budget, adapted online

    def step(self, prob_batch: np.ndarray, y_batch: np.ndarray) -> dict:
        """Decide approvals for one time cohort, observe outcomes, update the budget."""
        prob = np.asarray(prob_batch, dtype=np.float64)
        y = np.asarray(y_batch, dtype=int)
        t = self.cmap.threshold_for(self.budget)
        approve = prob <= t
        realised = float(y[approve].mean()) if approve.any() else 0.0
        # ACI feedback: overshoot risk -> shrink budget (approve fewer); undershoot -> grow it.
        self.budget = float(np.clip(self.budget + self.gamma * (self.alpha_target - realised), 1e-4, 0.5))
        return {
            "threshold": t,
            "n": int(len(y)),
            "n_approved": int(approve.sum()),
            "coverage": float(approve.mean()),
            "default_rate_among_approved": realised,
            "budget_next": self.budget,
        }


def run_static(cmap_prob, cmap_y, alpha_target, year_batches):
    """Baseline: a single fixed threshold calibrated once, applied to every future cohort."""
    cmap = _CalibrationMap(cmap_prob, cmap_y)
    t = cmap.threshold_for(alpha_target)
    rows = []
    for yr, (p, y) in year_batches:
        approve = np.asarray(p) <= t
        realised = float(np.asarray(y)[approve].mean()) if approve.any() else 0.0
        rows.append((yr, {"threshold": t, "n": int(len(y)), "coverage": float(approve.mean()),
                          "default_rate_among_approved": realised, "n_approved": int(approve.sum())}))
    return rows


def run_adaptive(cmap_prob, cmap_y, alpha_target, gamma, year_batches):
    sel = AdaptiveConformalSelector(cmap_prob, cmap_y, alpha_target=alpha_target, gamma=gamma)
    rows = []
    for yr, (p, y) in year_batches:
        rows.append((yr, sel.step(p, y)))
    return rows
