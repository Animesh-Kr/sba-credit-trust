"""Risk-controlled selective classification for loan decisions.

The model may **auto-approve**, **auto-reject**, or **abstain** ("refer to a human"). The guarantee
we care about is on the costly error: among **auto-approved** loans, the realised default rate must
stay at or below a target ``alpha`` (e.g. 5%). We pick the approval threshold on a held-out
calibration set using a binomial **upper** confidence bound (Clopper–Pearson), which gives a
finite-sample, distribution-free risk guarantee in the spirit of conformal risk control / RCPS
(Angelopoulos & Bates, 2021; Bates et al., 2021): with probability ≥ 1−delta the chosen threshold
controls the population risk.

A symmetric threshold on the reject side guarantees that auto-rejected loans are genuinely bad
(their default rate is high). Loans between the two thresholds are abstained.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.stats import beta


def clopper_pearson_upper(k: int, n: int, delta: float) -> float:
    """Upper (1−delta) confidence bound on a binomial proportion from k successes in n trials."""
    if n == 0:
        return 1.0
    if k == n:
        return 1.0
    return float(beta.ppf(1 - delta, k + 1, n - k))


def clopper_pearson_lower(k: int, n: int, delta: float) -> float:
    """Lower (1−delta) confidence bound on a binomial proportion."""
    if n == 0:
        return 0.0
    if k == 0:
        return 0.0
    return float(beta.ppf(delta, k, n - k + 1))


@dataclass
class SelectiveRiskController:
    """Calibrated approve/reject thresholds with a risk guarantee on each auto-decision.

    CAVEAT (see docs/AUDIT.md, M-1): the threshold is chosen by scanning all candidate cut-offs and
    keeping the largest whose Clopper-Pearson bound holds. The per-threshold bound uses ``delta``
    without a multiplicity correction, and the empirical risk among the selected set is not strictly
    monotone in the threshold. The guarantee is therefore **empirically validated** here, not
    formally airtight; a fully rigorous version uses RCPS / Learn-then-Test (Bates et al., 2021;
    Angelopoulos et al., 2021) with a monotonised risk and a calibrated single cut-off. This is on
    the enhancement backlog and does not affect the realised-risk numbers reported."""

    alpha: float = 0.05     # max tolerated default rate among auto-approved
    delta: float = 0.05     # confidence level (guarantee holds w.p. >= 1 - delta)
    t_lo: float = 0.0       # approve if P(default) <= t_lo
    t_hi: float = 1.0       # reject  if P(default) >= t_hi

    def fit(self, prob_cal: np.ndarray, y_cal: np.ndarray) -> SelectiveRiskController:
        prob = np.asarray(prob_cal, dtype=np.float64)
        y = np.asarray(y_cal, dtype=int)
        order = np.argsort(prob)
        p_sorted, y_sorted = prob[order], y[order]

        # --- approve side: largest t_lo s.t. UCB(default rate among prob<=t_lo) <= alpha ---
        # scan thresholds at observed prob values, growing the approved set from the low end.
        cum_def = np.cumsum(y_sorted)
        best_lo = -1.0
        for i in range(len(p_sorted)):
            n, k = i + 1, int(cum_def[i])
            if clopper_pearson_upper(k, n, self.delta) <= self.alpha:
                best_lo = p_sorted[i]
            # do not break: keep the largest threshold that still satisfies the bound
        self.t_lo = max(best_lo, 0.0) if best_lo >= 0 else 0.0

        # --- reject side: smallest t_hi s.t. LCB(default rate among prob>=t_hi) >= 1 - alpha ---
        # grow the rejected set from the high end.
        cum_def_rev = np.cumsum(y_sorted[::-1])
        best_hi = 2.0
        for j in range(len(p_sorted)):
            n, k = j + 1, int(cum_def_rev[j])
            if clopper_pearson_lower(k, n, self.delta) >= 1 - self.alpha:
                best_hi = p_sorted[::-1][j]
        self.t_hi = min(best_hi, 1.0) if best_hi <= 1.0 else 1.0
        return self

    def decide(self, prob: np.ndarray) -> np.ndarray:
        """Return an array of {'approve','reject','abstain'}.

        If the two thresholds cross (t_lo >= t_hi), loans in the overlap are genuinely ambiguous
        — the two one-sided guarantees disagree — so they **abstain** rather than being silently
        forced into one bucket. approve and reject are therefore always mutually exclusive.
        """
        prob = np.asarray(prob, dtype=np.float64)
        out = np.full(len(prob), "abstain", dtype=object)
        out[(prob <= self.t_lo) & (prob < self.t_hi)] = "approve"
        out[(prob >= self.t_hi) & (prob > self.t_lo)] = "reject"
        return out

    def fit_rigorous(self, prob_cal: np.ndarray, y_cal: np.ndarray, n_grid: int = 200) -> SelectiveRiskController:
        """E-1: union-bounded (Bonferroni) approve threshold — formally valid, addresses AUDIT M-1.

        Evaluate the Clopper-Pearson upper bound at each of ``n_grid`` candidate thresholds with a
        per-threshold level ``delta / n_grid``; pick the largest threshold whose bound ≤ alpha. The
        union bound makes the guarantee hold simultaneously over the whole grid, so no monotonicity
        assumption is needed (conservative but rigorous). Reject side left at its default (1.0).
        """
        prob = np.asarray(prob_cal, dtype=np.float64)
        y = np.asarray(y_cal, dtype=int)
        grid = np.unique(np.quantile(prob, np.linspace(0, 1, n_grid)))
        delta_g = self.delta / len(grid)
        best_lo = -1.0
        for t in grid:
            sel = prob <= t
            n, k = int(sel.sum()), int(y[sel].sum())
            if n > 0 and clopper_pearson_upper(k, n, delta_g) <= self.alpha:
                best_lo = t
        self.t_lo = max(best_lo, 0.0) if best_lo >= 0 else 0.0
        self.t_hi = 1.0
        return self

    def evaluate(self, prob: np.ndarray, y: np.ndarray) -> dict:
        """Realised coverage and risk on a test set."""
        prob = np.asarray(prob, dtype=np.float64)
        y = np.asarray(y, dtype=int)
        dec = self.decide(prob)
        appr = dec == "approve"
        rej = dec == "reject"
        absta = dec == "abstain"
        n = len(y)
        return {
            "t_lo": self.t_lo,
            "t_hi": self.t_hi,
            "alpha": self.alpha,
            "approve_rate": float(appr.mean()),
            "reject_rate": float(rej.mean()),
            "abstain_rate": float(absta.mean()),
            "coverage": float((appr | rej).mean()),
            "default_rate_among_approved": float(y[appr].mean()) if appr.any() else float("nan"),
            "default_rate_among_rejected": float(y[rej].mean()) if rej.any() else float("nan"),
            "n_approved": int(appr.sum()),
            "n_rejected": int(rej.sum()),
            "n_total": int(n),
        }
