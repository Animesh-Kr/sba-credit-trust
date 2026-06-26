"""Phase-A baseline runner -> reports/baseline_metrics.md.

Trains leakage-free Logistic Regression and XGBoost on a stratified hold-out and reports the
honest metric suite (PR-AUC, ROC-AUC, F1/precision/recall-minority, Brier, ECE). Optionally
quantifies the over-reporting caused by the reference demo's oversample-before-split leakage by
reproducing that flawed pipeline and showing the inflated test metric.
"""
from __future__ import annotations

import os
import time

from ..data import clean, split
from ..eval import metrics
from ..models import baselines

REPORT = os.path.join(os.path.dirname(__file__), "..", "..", "reports", "baseline_metrics.md")


def _fit_predict(model, X_tr, y_tr, X_te):
    model.fit(X_tr, y_tr)
    return model.predict_proba(X_te)[:, 1]


def run(
    raw_csv: str,
    include_disbursement: bool = False,
    test_size: float = 0.2,
    seed: int = 42,
    quantify_leakage: bool = True,
    nrows: int | None = None,
) -> str:
    t0 = time.time()
    X, y = clean.load_clean(raw_csv, include_disbursement=include_disbursement, nrows=nrows)
    train_idx, test_idx = split.stratified_split(y, test_size=test_size, seed=seed)
    X_tr, X_te = X.iloc[train_idx], X.iloc[test_idx]
    y_tr, y_te = y.iloc[train_idx], y.iloc[test_idx]

    results: dict[str, dict] = {}

    lr = baselines.logistic_regression(include_disbursement, seed)
    results["LogisticRegression"] = metrics.classification_report(
        y_te.to_numpy(), _fit_predict(lr, X_tr, y_tr, X_te)
    )

    xgb = baselines.xgboost(y_tr, include_disbursement, seed)
    results["XGBoost"] = metrics.classification_report(
        y_te.to_numpy(), _fit_predict(xgb, X_tr, y_tr, X_te)
    )

    # --- leakage demonstration: oversample BEFORE split (the wrong way) ---
    leak = None
    if quantify_leakage:
        Xo, yo = baselines.naive_oversample_before_split(X, y, seed=seed)
        tr_i, te_i = split.stratified_split(yo, test_size=test_size, seed=seed)
        xgb_leak = baselines.xgboost(yo.iloc[tr_i], include_disbursement, seed)
        p_leak = _fit_predict(xgb_leak, Xo.iloc[tr_i], yo.iloc[tr_i], Xo.iloc[te_i])
        leak = metrics.classification_report(yo.iloc[te_i].to_numpy(), p_leak)

    out = _write_report(results, leak, X, y, train_idx, test_idx, include_disbursement, time.time() - t0)
    print(f"[baseline] wrote {out}  ({time.time()-t0:.1f}s)")
    return out


def _fmt(d: dict) -> str:
    keys = ["pr_auc", "roc_auc", "f1_minority", "precision_minority", "recall_minority", "brier", "ece"]
    return " | ".join(f"{d[k]:.4f}" for k in keys)


def _write_report(results, leak, X, y, train_idx, test_idx, include_disb, secs) -> str:
    lines = ["# SBA Credit-Trust — Phase A Baseline (leakage-free)\n"]
    lines.append(f"- Samples: **{len(y):,}**  ·  features: **{X.shape[1]}**  ·  "
                 f"default prevalence: **{y.mean():.4f}**")
    lines.append(f"- Split: stratified hold-out — train **{len(train_idx):,}** / test **{len(test_idx):,}**")
    lines.append(f"- include_disbursement: **{include_disb}**  ·  runtime: **{secs:.1f}s**\n")
    lines.append("All metrics are on the held-out test set. Positive class = **default (CHGOFF)**. "
                 "Threshold = 0.5.\n")
    lines.append("| Model | PR-AUC | ROC-AUC | F1(def) | Prec(def) | Recall(def) | Brier | ECE |")
    lines.append("|---|---|---|---|---|---|---|---|")
    for name, d in results.items():
        lines.append(f"| {name} | {_fmt(d)} |")
    lines.append("")
    if leak is not None:
        lines.append("## Leakage demonstration (do NOT trust these numbers)\n")
        lines.append("Reproducing the reference demo's flaw — random oversampling of the minority "
                     "class **before** the train/test split, so duplicated default rows appear in "
                     "both. The inflated test metrics below are the over-reporting artefact our "
                     "leakage-free pipeline corrects:\n")
        lines.append("| Pipeline | PR-AUC | ROC-AUC | F1(def) | Prec(def) | Recall(def) | Brier | ECE |")
        lines.append("|---|---|---|---|---|---|---|---|")
        lines.append(f"| XGBoost (oversample-before-split, LEAKY) | {_fmt(leak)} |")
        honest = results["XGBoost"]["pr_auc"]
        lines.append(f"\n> Honest XGBoost PR-AUC = **{honest:.4f}**; leaky PR-AUC = "
                     f"**{leak['pr_auc']:.4f}** → over-report of "
                     f"**{leak['pr_auc']-honest:+.4f}** purely from pipeline order.\n")
    os.makedirs(os.path.dirname(os.path.abspath(REPORT)), exist_ok=True)
    out = os.path.abspath(REPORT)
    with open(out, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    return out


if __name__ == "__main__":
    import sys

    csv = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "data", "SBAnational.csv"))
    nrows = int(sys.argv[1]) if len(sys.argv) > 1 else None
    run(csv, nrows=nrows)
