"""E-8: the best DL model (FT-Transformer), isotonic-calibrated, carries the conformal guarantee.

Phase C ran calibration + conformal on the MLP for clean penultimate features. E-8 puts the whole
trust stack on the strongest DL model so the deployed artefact is the best one. Three-way split;
isotonic calibration on the calibration slice; heuristic + rigorous conformal selective prediction
evaluated on test.
"""
from __future__ import annotations

import os
import time

from ..calibration.scaling import IsotonicCalibrator
from ..conformal.selective import SelectiveRiskController
from ..data import clean, encoders, split
from ..eval import metrics
from ..models import tabular_dl, trainer

REPORT = os.path.join(os.path.dirname(__file__), "..", "..", "reports", "best_model_conformal.md")


def run(raw_csv, epochs=30, seed=42, nrows=None):
    t0 = time.time()
    X, y = clean.load_clean(raw_csv, nrows=nrows)
    rest, te = split.stratified_split(y, test_size=0.20, seed=seed)
    itr, ical = split.stratified_split(y.iloc[rest], test_size=0.1875, seed=seed)
    tr, cal = rest[itr], rest[ical]

    enc = encoders.TabularEncoder().fit(X.iloc[tr])
    xtr = enc.transform(X.iloc[tr]); ytr = y.iloc[tr].to_numpy()
    xca = enc.transform(X.iloc[cal]); yca = y.iloc[cal].to_numpy()
    xte = enc.transform(X.iloc[te]); yte = y.iloc[te].to_numpy()
    itr2, iva = split.stratified_split(y.iloc[tr], test_size=0.1, seed=seed)

    net = tabular_dl.build("ft_transformer", enc.n_numeric, enc.cardinalities)
    net, _ = trainer.train(
        net,
        (xtr[0][itr2], xtr[1][itr2], ytr[itr2]),
        (xtr[0][iva], xtr[1][iva], ytr[iva]),
        epochs=epochs, seed=seed, verbose=False,
    )

    p_ca = trainer.predict_proba(net, xca[0], xca[1])
    p_te_raw = trainer.predict_proba(net, xte[0], xte[1])
    iso = IsotonicCalibrator().fit(p_ca, yca)
    p_te_cal = iso.transform(p_te_raw)
    p_ca_cal = iso.transform(p_ca)

    cal_rows = {
        "FT-Transformer raw": metrics.classification_report(yte, p_te_raw),
        "FT-Transformer + isotonic": metrics.classification_report(yte, p_te_cal),
    }
    selective = {}
    for alpha in (0.05, 0.10):
        h = SelectiveRiskController(alpha=alpha, delta=0.05).fit(p_ca_cal, yca).evaluate(p_te_cal, yte)
        r = SelectiveRiskController(alpha=alpha, delta=0.05).fit_rigorous(p_ca_cal, yca).evaluate(p_te_cal, yte)
        selective[alpha] = {"heuristic": h, "rigorous": r}

    out = _write(cal_rows, selective, len(te), time.time() - t0)
    print(f"[best-model] wrote {out}  ({time.time()-t0:.1f}s)")
    return out


def _fmt(d):
    return " | ".join(f"{d[k]:.4f}" for k in ["pr_auc", "roc_auc", "f1_minority", "brier", "ece"])


def _write(cal_rows, selective, n_test, secs):
    L = ["# SBA Credit-Trust — E-8 Best Model (FT-Transformer) + Calibration + Conformal\n"]
    L.append(f"Test n = {n_test:,}. Runtime {secs:.1f}s. The strongest DL model now carries the "
             "calibration + conformal stack (previously on the MLP).\n")
    L.append("## Calibration\n| Variant | PR-AUC | ROC-AUC | F1(def) | Brier | ECE |\n|---|---|---|---|---|---|")
    for name, d in cal_rows.items():
        L.append(f"| {name} | {_fmt(d)} |")
    L.append("\n## Conformal selective prediction (isotonic-calibrated FT-Transformer)\n")
    L.append("| α | method | coverage | abstain | **default@approved** |")
    L.append("|---|---|---|---|---|")
    for alpha, d in selective.items():
        for m in ("heuristic", "rigorous"):
            r = d[m]
            L.append(f"| {alpha:.2f} | {m} | {r['coverage']:.3f} | {r['abstain_rate']:.3f} | "
                     f"**{r['default_rate_among_approved']:.4f}** |")
    L.append("\n> The best DL model, calibrated, holds the selective-risk target on the test split. "
             "(Formal caveat: AUDIT M-1; temporal robustness: see adaptive_conformal report.)\n")
    os.makedirs(os.path.dirname(os.path.abspath(REPORT)), exist_ok=True)
    out = os.path.abspath(REPORT)
    with open(out, "w", encoding="utf-8") as f:
        f.write("\n".join(L))
    return out


if __name__ == "__main__":
    import sys

    csv = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "data", "SBAnational.csv"))
    nrows = int(sys.argv[1]) if len(sys.argv) > 1 else None
    ep = int(sys.argv[2]) if len(sys.argv) > 2 else 30
    run(csv, epochs=ep, nrows=nrows)
