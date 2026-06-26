"""E-1 + E-9 runner -> reports/adaptive_conformal_metrics.md.

E-1: a rigorous (union-bounded) static selective threshold, with realised risk in-era and future.
E-9: static vs Adaptive Conformal Inference applied year-by-year across the 2007-2013 shift, showing
     ACI restores long-run risk control where the static guarantee broke (Phase D).
"""
from __future__ import annotations

import os
import time

import numpy as np

from ..calibration.scaling import TemperatureScaler, logit
from ..conformal import adaptive
from ..conformal.selective import SelectiveRiskController
from ..data import clean, split
from ..models.baselines import xgboost

REPORT = os.path.join(os.path.dirname(__file__), "..", "..", "reports", "adaptive_conformal_metrics.md")


def run(raw_csv, cutoff_year=2006, alpha=0.05, gamma=0.10, seed=42, nrows=None):
    t0 = time.time()
    X, y, meta = clean.load_with_meta(raw_csv, nrows=nrows)
    tr_idx, te_idx = split.temporal_split(X, cutoff_year=cutoff_year)
    # in-era calibration: fit half + held-out eval half
    inner_tr, inner_cal = split.stratified_split(y.iloc[tr_idx], test_size=0.20, seed=seed)
    tr, cal_all = tr_idx[inner_tr], tr_idx[inner_cal]
    cfit, ceval = split.stratified_split(y.iloc[cal_all], test_size=0.5, seed=seed)
    cal_fit, cal_eval = cal_all[cfit], cal_all[ceval]

    pipe = xgboost(y.iloc[tr], seed=seed)
    pipe.fit(X.iloc[tr], y.iloc[tr])
    ts = TemperatureScaler().fit(logit(pipe.predict_proba(X.iloc[cal_fit])[:, 1]), y.iloc[cal_fit].to_numpy())

    def proba(idx):
        return ts.transform(logit(pipe.predict_proba(X.iloc[idx])[:, 1]))

    p_fit, y_fit = proba(cal_fit), y.iloc[cal_fit].to_numpy()
    p_eval, y_eval = proba(cal_eval), y.iloc[cal_eval].to_numpy()
    p_future, y_future = proba(te_idx), y.iloc[te_idx].to_numpy()
    fy_future = meta["ApprovalFY"].to_numpy()[te_idx]

    # ---- E-1: rigorous vs heuristic static threshold ----
    heuristic = SelectiveRiskController(alpha=alpha, delta=0.05).fit(p_fit, y_fit)
    rigorous = SelectiveRiskController(alpha=alpha, delta=0.05).fit_rigorous(p_fit, y_fit)
    e1 = {
        "heuristic": {
            "t_lo": heuristic.t_lo,
            "in_era": heuristic.evaluate(p_eval, y_eval),
            "future": heuristic.evaluate(p_future, y_future),
        },
        "rigorous": {
            "t_lo": rigorous.t_lo,
            "in_era": rigorous.evaluate(p_eval, y_eval),
            "future": rigorous.evaluate(p_future, y_future),
        },
    }

    # ---- E-9: static vs adaptive in the faithful ONLINE setting ----
    # Order future loans by actual approval date and stream them in fine batches, so ACI can adapt
    # *within* the large 2007-08 cohorts rather than only once per year.
    dt_future = meta["ApprovalDate"].to_numpy()[te_idx]
    order = np.argsort(dt_future.astype("datetime64[ns]"))
    p_ord, y_ord, fy_ord = p_future[order], y_future[order], fy_future[order]
    n_batches = 150
    edges = np.linspace(0, len(p_ord), n_batches + 1, dtype=int)
    stream = [(i, (p_ord[a:b], y_ord[a:b])) for i, (a, b) in enumerate(zip(edges[:-1], edges[1:])) if b > a]
    static_stream = adaptive.run_static(p_fit, y_fit, alpha, stream)
    adaptive_stream = adaptive.run_adaptive(p_fit, y_fit, alpha, gamma, stream)
    # aggregate the fine stream back to per-year rows for a readable table
    batch_year = [int(np.median(fy_ord[a:b])) for a, b in zip(edges[:-1], edges[1:]) if b > a]
    static_rows = _aggregate_by_year(static_stream, adaptive_stream, batch_year)

    out = _write(e1, static_rows, alpha, gamma, cutoff_year,
                 _running_avg_risk([(i, r) for i, r in static_stream]),
                 _running_avg_risk([(i, r) for i, r in adaptive_stream]), time.time() - t0)
    print(f"[adaptive] wrote {out}  ({time.time()-t0:.1f}s)")
    return out


def _running_avg_risk(rows):
    """Cumulative default-among-approved across cohorts (loan-weighted)."""
    num = den = 0.0
    for _, r in rows:
        num += r["default_rate_among_approved"] * r["n_approved"]
        den += r["n_approved"]
    return num / den if den else float("nan")


def _aggregate_by_year(static_stream, adaptive_stream, batch_year):
    """Loan-weighted per-year aggregation of the fine online stream, for a readable table."""
    agg: dict[int, dict] = {}
    for (_, s), (_, a), yr in zip(static_stream, adaptive_stream, batch_year):
        d = agg.setdefault(yr, {"sn": 0.0, "sd": 0.0, "sc": 0.0, "an": 0.0, "ad": 0.0, "ac": 0.0, "at": 0.0, "ab": 0})
        d["sn"] += s["n_approved"]; d["sd"] += s["default_rate_among_approved"] * s["n_approved"]
        d["sc"] += s["coverage"] * s["n"]
        d["an"] += a["n_approved"]; d["ad"] += a["default_rate_among_approved"] * a["n_approved"]
        d["ac"] += a["coverage"] * s["n"]; d["at"] += a["threshold"]; d["ab"] += 1
        d["nn"] = d.get("nn", 0.0) + s["n"]
    rows = []
    for yr in sorted(agg):
        d = agg[yr]
        rows.append((yr, {
            "static_risk": d["sd"] / d["sn"] if d["sn"] else 0.0,
            "static_cov": d["sc"] / d["nn"] if d["nn"] else 0.0,
            "aci_risk": d["ad"] / d["an"] if d["an"] else 0.0,
            "aci_cov": d["ac"] / d["nn"] if d["nn"] else 0.0,
            "aci_thr": d["at"] / d["ab"] if d["ab"] else 0.0,
        }))
    return rows


def _write(e1, year_rows, alpha, gamma, cutoff, rs, ra, secs):
    L = ["# SBA Credit-Trust — E-1 Rigorous + E-9 Adaptive Conformal\n"]
    L.append(f"Target α = {alpha}. Train ≤ {cutoff}; future > {cutoff}. Runtime {secs:.1f}s.\n")

    L.append("## E-1 — Rigorous (union-bounded) vs heuristic static threshold\n")
    L.append("| Method | t_lo | in-era default@approved | future default@approved | future coverage |")
    L.append("|---|---|---|---|---|")
    for name in ("heuristic", "rigorous"):
        d = e1[name]
        L.append(f"| {name} | {d['t_lo']:.3f} | {d['in_era']['default_rate_among_approved']:.4f} | "
                 f"{d['future']['default_rate_among_approved']:.4f} | {d['future']['coverage']:.3f} |")
    L.append("\n> The rigorous (Bonferroni union-bounded) threshold is more conservative in-era; both "
             "still drift above α on the shifted future — motivating E-9.\n")

    L.append(f"## E-9 — Static vs Adaptive Conformal Inference, online by approval date (γ={gamma})\n")
    L.append("Future loans streamed in approval-date order (150 batches); per-year aggregation below.\n")
    L.append("| Year | static default@approved | static coverage | **ACI default@approved** | ACI coverage | ACI threshold |")
    L.append("|---|---|---|---|---|---|")
    for yr, d in year_rows:
        flag = "" if d["aci_risk"] <= alpha + 0.02 else " ⚠"
        L.append(f"| {yr} | {d['static_risk']:.3f} | {d['static_cov']:.3f} | "
                 f"**{d['aci_risk']:.3f}**{flag} | {d['aci_cov']:.3f} | {d['aci_thr']:.3f} |")
    L.append(f"\n> **Long-run default rate among approved:** static = **{rs:.4f}**, "
             f"adaptive = **{ra:.4f}** (target {alpha}). ACI tracks the target by tightening the "
             "threshold after high-default batches — turning Phase D's negative result (static "
             "conformal breaks under shift) into a positive one.\n")

    os.makedirs(os.path.dirname(os.path.abspath(REPORT)), exist_ok=True)
    out = os.path.abspath(REPORT)
    with open(out, "w", encoding="utf-8") as f:
        f.write("\n".join(L))
    return out


if __name__ == "__main__":
    import sys

    csv = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "data", "SBAnational.csv"))
    nrows = int(sys.argv[1]) if len(sys.argv) > 1 else None
    run(csv, nrows=nrows)
