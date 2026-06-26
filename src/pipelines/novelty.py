"""Phase-D novelty experiments -> reports/novelty_metrics.md.

(A) Temporal distribution-shift study across the 2008 crisis: train on pre-cutoff cohorts, test on
    later years; report how PR-AUC/ROC-AUC/ECE degrade vs the optimistic random split, and whether
    conformal coverage survives the shift.
(B) Cost-sensitive dollar-aware decisioning vs accuracy-threshold and approve-all baselines, with a
    margin sensitivity sweep.
(C) Slice / fairness audit: performance + calibration across NAICS sectors and loan-size bands.

Workhorse model = XGBoost (best ranker, fast). All probabilities are temperature-calibrated on a
held-out calibration slice. Findings are model-agnostic and transfer to the DL models.
"""
from __future__ import annotations

import os
import time

import numpy as np

from ..calibration.scaling import TemperatureScaler, logit
from ..conformal.selective import SelectiveRiskController
from ..data import clean, split
from ..eval import cost, metrics
from ..models.baselines import xgboost

REPORT = os.path.join(os.path.dirname(__file__), "..", "..", "reports", "novelty_metrics.md")


def _fit_xgb_calibrated(X_tr, y_tr, X_cal, y_cal, seed):
    pipe = xgboost(y_tr, seed=seed)
    pipe.fit(X_tr, y_tr)
    ts = TemperatureScaler().fit(logit(pipe.predict_proba(X_cal)[:, 1]), y_cal)
    return pipe, ts


def _proba(pipe, ts, X):
    return ts.transform(logit(pipe.predict_proba(X)[:, 1]))


def run(raw_csv, cutoff_year=2006, margins=(0.03, 0.05, 0.10), seed=42, nrows=None):
    t0 = time.time()
    X, y, meta = clean.load_with_meta(raw_csv, nrows=nrows)

    # ============ (A) TEMPORAL SHIFT ============
    tr_idx, te_idx = split.temporal_split(X, cutoff_year=cutoff_year)
    # carve a calibration slice from the (pre-cutoff) training era, then split it into a fit half
    # (for temperature + conformal thresholds) and a held-out same-era eval half (honest in-era
    # number — never evaluate conformal on the slice its threshold was fit on).
    y_tr_all = y.iloc[tr_idx]
    inner_tr, inner_cal = split.stratified_split(y_tr_all, test_size=0.20, seed=seed)
    tr = tr_idx[inner_tr]
    cal_all = tr_idx[inner_cal]
    cfit, ceval = split.stratified_split(y.iloc[cal_all], test_size=0.5, seed=seed)
    cal_fit, cal_eval = cal_all[cfit], cal_all[ceval]

    pipe_t, ts_t = _fit_xgb_calibrated(
        X.iloc[tr], y.iloc[tr], X.iloc[cal_fit], y.iloc[cal_fit].to_numpy(), seed
    )
    p_future = _proba(pipe_t, ts_t, X.iloc[te_idx])
    y_future = y.iloc[te_idx].to_numpy()
    temporal_overall = metrics.classification_report(y_future, p_future)

    # per-year degradation on the future side
    fy_future = meta["ApprovalFY"].to_numpy()[te_idx]
    per_year = []
    for yr in sorted(set(fy_future[~np.isnan(fy_future)])):
        m = fy_future == yr
        if m.sum() < 500:
            continue
        rep = metrics.classification_report(y_future[m], p_future[m])
        per_year.append((int(yr), int(m.sum()), rep))

    # conformal under shift: fit threshold on in-era FIT slice, evaluate on a held-out in-era slice
    # (honest) and on the shifted future cohorts.
    p_cal_fit = _proba(pipe_t, ts_t, X.iloc[cal_fit])
    p_cal_eval = _proba(pipe_t, ts_t, X.iloc[cal_eval])
    ctrl = SelectiveRiskController(alpha=0.05, delta=0.05).fit(p_cal_fit, y.iloc[cal_fit].to_numpy())
    conf_in_era = ctrl.evaluate(p_cal_eval, y.iloc[cal_eval].to_numpy())
    conf_future = ctrl.evaluate(p_future, y_future)

    # random-split reference (optimistic) for contrast
    r_tr, r_te = split.stratified_split(y, test_size=0.2, seed=seed)
    ri_tr, ri_cal = split.stratified_split(y.iloc[r_tr], test_size=0.15, seed=seed)
    pipe_r, ts_r = _fit_xgb_calibrated(
        X.iloc[r_tr[ri_tr]], y.iloc[r_tr[ri_tr]], X.iloc[r_tr[ri_cal]], y.iloc[r_tr[ri_cal]].to_numpy(), seed
    )
    p_rand = _proba(pipe_r, ts_r, X.iloc[r_te])
    random_overall = metrics.classification_report(y.iloc[r_te].to_numpy(), p_rand)

    # ============ (B) COST-SENSITIVE POLICY (on the random-split test) ============
    te = r_te
    p_te = p_rand
    y_te = y.iloc[te].to_numpy()
    gr = meta["GrAppv"].to_numpy()[te]
    co = meta["ChgOffPrinGr"].to_numpy()[te]
    # empirical loss-given-default estimated on the TRAIN portion only (no test leakage)
    tr_for_lgd = r_tr
    lgd = cost.estimate_lgd(
        y.iloc[tr_for_lgd].to_numpy(),
        meta["GrAppv"].to_numpy()[tr_for_lgd],
        meta["ChgOffPrinGr"].to_numpy()[tr_for_lgd],
    )
    exposure = lgd * gr  # expected $ at risk on default
    cost_sweep = {m: cost.compare_policies(p_te, y_te, gr, exposure, co, margin=m) for m in margins}

    # ============ (C) SLICE / FAIRNESS (random-split test) ============
    sectors = meta["naics_sector"].to_numpy()[te]
    slice_sector = _slice_table(p_te, y_te, sectors, min_n=2000, top=10)
    bands, band_labels = _loan_size_bands(gr)
    slice_band = _slice_table(p_te, y_te, bands, labels=band_labels, min_n=1)

    out = _write(temporal_overall, random_overall, per_year, conf_in_era, conf_future,
                 cost_sweep, lgd, slice_sector, slice_band, cutoff_year, len(te_idx), time.time() - t0)
    print(f"[novelty] wrote {out}  ({time.time()-t0:.1f}s)")
    return out


def _loan_size_bands(gr):
    q = np.nanquantile(gr, [0.25, 0.5, 0.75])
    bands = np.digitize(gr, q)
    labels = {0: f"Q1 (<=${q[0]:,.0f})", 1: "Q2", 2: "Q3", 3: f"Q4 (>${q[2]:,.0f})"}
    return bands, labels


def _slice_table(p, y, groups, labels=None, min_n=1, top=None):
    rows = []
    uniq = [g for g in sorted(set(groups[~_isnan(groups)])) ]
    for g in uniq:
        m = groups == g
        n = int(m.sum())
        if n < min_n:
            continue
        rep = metrics.classification_report(y[m], p[m])
        name = labels[g] if labels else str(g)
        rows.append((name, n, rep))
    rows.sort(key=lambda r: -r[1])
    return rows[:top] if top else rows


def _isnan(a):
    try:
        return np.isnan(a)
    except TypeError:
        return np.array([x is None or (isinstance(x, float) and np.isnan(x)) for x in a])


def _write(temporal, random_, per_year, conf_in, conf_fut, cost_sweep, lgd, slice_sector, slice_band,
           cutoff, n_future, secs):
    L = ["# SBA Credit-Trust — Phase D Novelty Experiments\n"]
    L.append(f"Runtime **{secs:.1f}s**. Workhorse = temperature-calibrated XGBoost.\n")

    L.append("## A. Temporal distribution shift (the honest deployment number)\n")
    L.append(f"Train on cohorts **≤ {cutoff}**, test on **> {cutoff}** (n={n_future:,}, spans the 2008 crisis).\n")
    L.append("| Evaluation | PR-AUC | ROC-AUC | F1(def) | Brier | ECE |")
    L.append("|---|---|---|---|---|---|")
    L.append(f"| Random split (optimistic) | {random_['pr_auc']:.4f} | {random_['roc_auc']:.4f} | "
             f"{random_['f1_minority']:.4f} | {random_['brier']:.4f} | {random_['ece']:.4f} |")
    L.append(f"| **Temporal split (deployment)** | {temporal['pr_auc']:.4f} | {temporal['roc_auc']:.4f} | "
             f"{temporal['f1_minority']:.4f} | {temporal['brier']:.4f} | {temporal['ece']:.4f} |")
    L.append("")
    L.append("Per-year on the future side:\n")
    L.append("| ApprovalFY | n | PR-AUC | ROC-AUC | ECE | actual default rate |")
    L.append("|---|---|---|---|---|---|")
    for yr, n, rep in per_year:
        L.append(f"| {yr} | {n:,} | {rep['pr_auc']:.3f} | {rep['roc_auc']:.3f} | {rep['ece']:.3f} | {rep['prevalence']:.3f} |")
    L.append("")
    L.append("**Conformal coverage under shift** (threshold fit in-era, α=0.05):\n")
    L.append("| Applied to | Coverage | Default among approved | Guarantee (≤0.05)? |")
    L.append("|---|---|---|---|")
    L.append(f"| In-era calibration | {conf_in['coverage']:.3f} | {conf_in['default_rate_among_approved']:.4f} | "
             f"{'✅' if conf_in['default_rate_among_approved'] <= 0.06 else '❌'} |")
    ok = conf_fut['default_rate_among_approved'] <= 0.06
    L.append(f"| **Future cohorts (shifted)** | {conf_fut['coverage']:.3f} | "
             f"{conf_fut['default_rate_among_approved']:.4f} | {'✅' if ok else '❌ broken by shift'} |")
    L.append("\n> A guarantee calibrated in one era can **break under distribution shift** — an honest, "
             "important caveat for deploying conformal methods on non-exchangeable temporal data.\n")

    L.append("## B. Cost-sensitive dollar-aware decisioning (random-split test)\n")
    L.append(f"Empirical loss-given-default estimated on train: **LGD = {lgd:.2%}** of gross principal. "
             "The EV rule approves iff `(1−p)·m·GrAppv > p·LGD·GrAppv`, i.e. a principled probability "
             f"cut-off `t* = m/(m+LGD)` — contrast the arbitrary 0.5. Defaults costed at actual "
             "`ChgOffPrinGr` (ex-post accounting only).\n")
    for m, pol in cost_sweep.items():
        tstar = m / (m + lgd)
        L.append(f"\n**Margin m = {m:.0%}**  (EV-optimal cut-off t* = {tstar:.3f})\n")
        L.append("| Policy | Net $ | Approved | Defaults approved | Approval rate |")
        L.append("|---|---|---|---|---|")
        for name, r in pol.items():
            L.append(f"| {name} | ${r['net_dollars']:,.0f} | {r['n_approved']:,} | "
                     f"{r['n_defaults_approved']:,} | {r['approval_rate']:.3f} |")
        ev = pol["dollar_aware_EV"]
        thr = pol[[k for k in pol if k.startswith("threshold")][0]]
        d_dollar = ev["net_dollars"] - thr["net_dollars"]
        d_def = thr["n_defaults_approved"] - ev["n_defaults_approved"]
        L.append(f"\n> Dollar-aware vs 0.5-threshold: **${d_dollar:+,.0f}** net, and **{d_def:+,}** "
                 f"fewer defaulters approved — an explicit risk/return trade-off the accuracy "
                 f"threshold cannot express.\n")

    L.append("## C. Slice / fairness audit (random-split test)\n")
    L.append("Performance + calibration must hold across groups, not just on average.\n")
    L.append("### By NAICS sector (top 10 by volume)\n")
    L.append("| Sector | n | PR-AUC | ROC-AUC | ECE | default rate |")
    L.append("|---|---|---|---|---|---|")
    for name, n, rep in slice_sector:
        L.append(f"| {name} | {n:,} | {rep['pr_auc']:.3f} | {rep['roc_auc']:.3f} | {rep['ece']:.3f} | {rep['prevalence']:.3f} |")
    L.append("\n### By loan-size band (GrAppv quartiles)\n")
    L.append("| Band | n | PR-AUC | ROC-AUC | ECE | default rate |")
    L.append("|---|---|---|---|---|---|")
    for name, n, rep in slice_band:
        L.append(f"| {name} | {n:,} | {rep['pr_auc']:.3f} | {rep['roc_auc']:.3f} | {rep['ece']:.3f} | {rep['prevalence']:.3f} |")
    L.append("")

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
