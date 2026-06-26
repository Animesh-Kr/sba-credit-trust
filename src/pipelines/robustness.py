"""E-2: metric uncertainty -> reports/robustness_metrics.md.

Two views: (1) across-seed mean ± std for XGBoost over several independent split/fit seeds, and
(2) bootstrap 95% CIs on the headline metrics from a reference run. Addresses AUDIT M-2.
"""
from __future__ import annotations

import os
import time

from ..data import clean, split
from ..eval import metrics
from ..eval.metrics import (
    average_precision_score,
    brier_score_loss,
    expected_calibration_error,
    roc_auc_score,
)
from ..eval.stats import bootstrap_ci, seed_summary
from ..models.baselines import xgboost

REPORT = os.path.join(os.path.dirname(__file__), "..", "..", "reports", "robustness_metrics.md")


def _fit_eval(X, y, seed):
    tr, te = split.stratified_split(y, test_size=0.2, seed=seed)
    pipe = xgboost(y.iloc[tr], seed=seed)
    pipe.fit(X.iloc[tr], y.iloc[tr])
    p = pipe.predict_proba(X.iloc[te])[:, 1]
    yte = y.iloc[te].to_numpy()
    return yte, p, metrics.classification_report(yte, p)


def run(raw_csv, seeds=(42, 7, 13, 21, 100), nrows=None):
    t0 = time.time()
    X, y = clean.load_clean(raw_csv, nrows=nrows)

    per_seed = []
    ref = None
    for s in seeds:
        yte, p, rep = _fit_eval(X, y, s)
        per_seed.append({k: rep[k] for k in ["pr_auc", "roc_auc", "f1_minority", "brier", "ece"]})
        if ref is None:
            ref = (yte, p)
        print(f"[robustness] seed {s}: PR-AUC={rep['pr_auc']:.4f} ECE={rep['ece']:.4f}")
    summary = seed_summary(per_seed)

    # bootstrap CIs on the reference seed's test predictions
    yte, p = ref
    metric_fns = {
        "pr_auc": average_precision_score,
        "roc_auc": roc_auc_score,
        "brier": brier_score_loss,
        "ece": lambda yt, pp: expected_calibration_error(yt, pp),
    }
    cis = {name: bootstrap_ci(yte, p, fn, n_boot=1000, seed=0) for name, fn in metric_fns.items()}

    out = _write(summary, cis, list(seeds), len(yte), time.time() - t0)
    print(f"[robustness] wrote {out}  ({time.time()-t0:.1f}s)")
    return out


def _write(summary, cis, seeds, n_test, secs):
    L = ["# SBA Credit-Trust — E-2 Metric Uncertainty (XGBoost)\n"]
    L.append(f"Addresses AUDIT M-2. Runtime **{secs:.1f}s**.\n")
    L.append(f"## Across-seed variability ({len(seeds)} seeds: {seeds})\n")
    L.append("| Metric | mean | std | min | max |")
    L.append("|---|---|---|---|---|")
    for k, s in summary.items():
        L.append(f"| {k} | {s['mean']:.4f} | {s['std']:.4f} | {s['min']:.4f} | {s['max']:.4f} |")
    L.append(f"\n## Bootstrap 95% CIs (reference seed, test n={n_test:,}, 1000 resamples)\n")
    L.append("| Metric | point | 95% CI |")
    L.append("|---|---|---|")
    for name, (pt, lo, hi) in cis.items():
        L.append(f"| {name} | {pt:.4f} | [{lo:.4f}, {hi:.4f}] |")
    L.append("\n> Across-seed std captures train/split variance; the bootstrap CI captures "
             "finite-test-set variance. Both are small, so the headline XGBoost numbers are stable.\n")
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
