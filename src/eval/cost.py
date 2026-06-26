"""Cost-sensitive, dollar-aware loan decisioning.

Accuracy is the wrong objective when a missed default costs far more than a declined good loan.
This module turns a calibrated default probability into an **expected-monetary-value** decision and
scores the *realised dollars* of different policies on held-out data.

Economics (assumptions stated explicitly, with sensitivity analysis):
* Approving a loan that is **paid** earns a margin ``m`` on the gross approved amount: ``m * GrAppv``.
* Approving a loan that **defaults** loses the realised charged-off principal ``ChgOffPrinGr``
  (used only for ex-post accounting — it is post-outcome and never a model feature).
* At decision time the *expected* loss-given-default is proxied by the SBA-guaranteed amount
  ``SBA_Appv`` (the dollars actually at risk), which is known at approval.

Decision rule (expected value > 0): approve iff ``(1 - p) * m * GrAppv > p * SBA_Appv``.
"""
from __future__ import annotations

import numpy as np


def expected_value_decision(p, gr_appv, exposure, margin: float) -> np.ndarray:
    """Boolean approve mask from the expected-value rule (known-at-approval quantities only).

    ``exposure`` is the expected dollars at risk if the loan defaults — i.e. LGD × principal.
    Approve iff expected gain on a paying loan exceeds expected loss on a defaulting one.
    """
    p = np.asarray(p, dtype=np.float64)
    expected_gain = (1 - p) * margin * np.asarray(gr_appv, dtype=np.float64)
    expected_loss = p * np.asarray(exposure, dtype=np.float64)
    return expected_gain > expected_loss


def estimate_lgd(y, gr_appv, chgoff_prin) -> float:
    """Empirical loss-given-default as a fraction of gross approved principal, from training data.

    LGD = (total principal charged off on defaults) / (total gross approved on those defaults).
    """
    y = np.asarray(y, dtype=int)
    gr = np.asarray(gr_appv, dtype=np.float64)
    co = np.asarray(chgoff_prin, dtype=np.float64)
    defaulted = y == 1
    denom = gr[defaulted].sum()
    return float(co[defaulted].sum() / denom) if denom > 0 else 1.0


def realised_dollars(approve_mask, y, gr_appv, chgoff_prin, margin: float) -> dict:
    """Realised $ outcome of a policy, using actual labels and charged-off principal (ex-post)."""
    approve = np.asarray(approve_mask, dtype=bool)
    y = np.asarray(y, dtype=int)
    gr = np.asarray(gr_appv, dtype=np.float64)
    co = np.asarray(chgoff_prin, dtype=np.float64)

    paid = approve & (y == 0)
    defaulted = approve & (y == 1)
    profit = margin * gr[paid].sum()
    loss = co[defaulted].sum()
    return {
        "net_dollars": float(profit - loss),
        "profit_from_paid": float(profit),
        "loss_from_defaults": float(loss),
        "n_approved": int(approve.sum()),
        "n_defaults_approved": int(defaulted.sum()),
        "approval_rate": float(approve.mean()),
    }


def compare_policies(p, y, gr_appv, exposure, chgoff_prin, margin: float, threshold: float = 0.5) -> dict:
    """Realised $ for approve-all, accuracy-threshold, and the dollar-aware EV policy.

    ``exposure`` is the per-loan expected dollars at risk (LGD × principal) used by the EV rule.
    """
    p = np.asarray(p, dtype=np.float64)
    policies = {
        "approve_all": np.ones(len(p), dtype=bool),
        f"threshold_{threshold:g}": p < threshold,
        "dollar_aware_EV": expected_value_decision(p, gr_appv, exposure, margin),
    }
    return {
        name: realised_dollars(mask, y, gr_appv, chgoff_prin, margin)
        for name, mask in policies.items()
    }
