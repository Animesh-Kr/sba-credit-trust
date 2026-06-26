"""Phase-B runner: tabular DL (MLP+embeddings, FT-Transformer) -> reports/deep_metrics.md.

Reuses the exact Phase-A leakage-safe stratified split and metric suite so DL numbers are
directly comparable to the XGBoost baseline. The training fold is further split into train/val
for early stopping; encoders are fit on the train fold only.
"""
from __future__ import annotations

import os
import time

from ..data import clean, encoders, split
from ..eval import metrics
from ..models import tabular_dl, trainer

REPORT = os.path.join(os.path.dirname(__file__), "..", "..", "reports", "deep_metrics.md")


def _prep(X, y, train_idx, test_idx, include_disb, seed):
    # carve a validation set out of the training fold (stratified)
    y_tr_full = y.iloc[train_idx]
    inner_tr, inner_val = split.stratified_split(y_tr_full, test_size=0.1, seed=seed)
    tr = train_idx[inner_tr]
    va = train_idx[inner_val]

    enc = encoders.TabularEncoder(include_disbursement=include_disb).fit(X.iloc[tr])
    xtr_n, xtr_c = enc.transform(X.iloc[tr])
    xva_n, xva_c = enc.transform(X.iloc[va])
    xte_n, xte_c = enc.transform(X.iloc[test_idx])
    return (
        enc,
        (xtr_n, xtr_c, y.iloc[tr].to_numpy()),
        (xva_n, xva_c, y.iloc[va].to_numpy()),
        (xte_n, xte_c, y.iloc[test_idx].to_numpy()),
    )


def run(
    raw_csv: str,
    models=("mlp", "ft_transformer"),
    include_disbursement: bool = False,
    test_size: float = 0.2,
    epochs: int = 30,
    seed: int = 42,
    nrows: int | None = None,
) -> str:
    t0 = time.time()
    X, y = clean.load_clean(raw_csv, include_disbursement=include_disbursement, nrows=nrows)
    train_idx, test_idx = split.stratified_split(y, test_size=test_size, seed=seed)
    enc, train_data, val_data, test_data = _prep(X, y, train_idx, test_idx, include_disbursement, seed)
    xte_n, xte_c, yte = test_data

    results: dict[str, dict] = {}
    for name in models:
        print(f"[deep] training {name} ...")
        net = tabular_dl.build(name, enc.n_numeric, enc.cardinalities)
        net, best_val = trainer.train(net, train_data, val_data, epochs=epochs, seed=seed)
        proba = trainer.predict_proba(net, xte_n, xte_c)
        rep = metrics.classification_report(yte, proba)
        rep["val_pr_auc"] = float(best_val)
        results[name] = rep

    out = _write(results, X, y, train_idx, test_idx, include_disbursement, time.time() - t0)
    print(f"[deep] wrote {out}  ({time.time()-t0:.1f}s)")
    return out


_DISPLAY = {"mlp": "MLP + embeddings", "ft_transformer": "FT-Transformer"}


def _fmt(d):
    keys = ["pr_auc", "roc_auc", "f1_minority", "precision_minority", "recall_minority", "brier", "ece"]
    return " | ".join(f"{d[k]:.4f}" for k in keys)


def _write(results, X, y, train_idx, test_idx, include_disb, secs):
    lines = ["# SBA Credit-Trust — Phase B Tabular Deep Learning\n"]
    lines.append(f"- Samples: **{len(y):,}**  ·  features: **{X.shape[1]}**  ·  "
                 f"default prevalence: **{y.mean():.4f}**")
    lines.append(f"- Split: same stratified hold-out as Phase A — test **{len(test_idx):,}**")
    lines.append(f"- include_disbursement: **{include_disb}**  ·  runtime: **{secs:.1f}s**\n")
    lines.append("Held-out test metrics. Positive class = **default (CHGOFF)**. Threshold = 0.5. "
                 "Focal loss (α=0.75, γ=2), early stop on val PR-AUC.\n")
    lines.append("| Model | PR-AUC | ROC-AUC | F1(def) | Prec(def) | Recall(def) | Brier | ECE |")
    lines.append("|---|---|---|---|---|---|---|---|")
    lines.append("| _XGBoost (Phase A ref)_ | 0.9139 | 0.9785 | 0.8200 | 0.7316 | 0.9328 | 0.0533 | 0.0741 |")
    for name, d in results.items():
        lines.append(f"| {_DISPLAY.get(name, name)} | {_fmt(d)} |")
    lines.append("")
    lines.append("> Per the project thesis, DL is **not** expected to beat XGBoost on accuracy on "
                 "tabular data (Grinsztajn et al. 2022). These DL models are the substrate for the "
                 "Phase-C trust layer (calibration, MC-Dropout uncertainty, Mahalanobis OOD, "
                 "conformal selective prediction) — the actual contribution.\n")
    os.makedirs(os.path.dirname(os.path.abspath(REPORT)), exist_ok=True)
    out = os.path.abspath(REPORT)
    with open(out, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    return out


if __name__ == "__main__":
    import sys

    csv = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "data", "SBAnational.csv"))
    nrows = int(sys.argv[1]) if len(sys.argv) > 1 else None
    ep = int(sys.argv[2]) if len(sys.argv) > 2 else 30
    run(csv, epochs=ep, nrows=nrows)
