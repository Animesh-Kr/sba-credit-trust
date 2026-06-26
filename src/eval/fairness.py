"""E-10: group fairness metrics for the approve/deny decision.

Beyond per-slice performance (Phase D), this quantifies *disparity* across protected-ish groups
(geography, loan size, industry). Favourable outcome = loan approved (predicted non-default).

Metrics per group, plus the max-min gap across groups:
* approval_rate          — demographic parity if equal across groups
* approval_rate_good     — P(approved | truly paid) — equal-opportunity (favourable class)
* default_catch_rate     — P(rejected | truly default) — protects the lender symmetrically
* ece                    — calibration must hold within each group, not just overall
"""
from __future__ import annotations

import numpy as np

from .metrics import expected_calibration_error


def group_metrics(y, p, groups, threshold: float = 0.5, min_n: int = 500):
    """Per-group decision + calibration metrics. Approve iff p < threshold (predicted non-default)."""
    y = np.asarray(y, dtype=int)
    p = np.asarray(p, dtype=np.float64)
    groups = np.asarray(groups, dtype=object)
    approve = p < threshold
    rows = {}
    for g in sorted({v for v in groups if v is not None and not (isinstance(v, float) and np.isnan(v))}):
        m = groups == g
        if m.sum() < min_n:
            continue
        yg, pg, ag = y[m], p[m], approve[m]
        good, bad = yg == 0, yg == 1
        rows[g] = {
            "n": int(m.sum()),
            "base_default_rate": float(yg.mean()),
            "approval_rate": float(ag.mean()),
            "approval_rate_good": float(ag[good].mean()) if good.any() else float("nan"),
            "default_catch_rate": float((~ag[bad]).mean()) if bad.any() else float("nan"),
            "ece": float(expected_calibration_error(yg, pg)),
        }
    return rows


def disparity(rows: dict) -> dict:
    """Max-min gap across groups for each metric (0 = perfectly fair)."""
    keys = ["approval_rate", "approval_rate_good", "default_catch_rate", "ece"]
    out = {}
    for k in keys:
        vals = [r[k] for r in rows.values() if not np.isnan(r[k])]
        out[f"{k}_gap"] = float(max(vals) - min(vals)) if vals else float("nan")
    return out
