"""E-4: real out-of-distribution benchmark via leave-one-sector-out -> reports/ood_metrics.md.

Addresses AUDIT M-4 (Phase-C OOD used a synthetic feature-permutation). Here a whole NAICS sector
is held out of training; at test time the Mahalanobis detector should flag loans from that unseen
industry as OOD relative to in-distribution test loans. We report AUROC for the real benchmark and,
for context, the synthetic permutation baseline.
"""
from __future__ import annotations

import os
import time

import numpy as np
from sklearn.metrics import roc_auc_score

from ..data import clean, encoders, split
from ..models import tabular_dl, trainer, uncertainty

REPORT = os.path.join(os.path.dirname(__file__), "..", "..", "reports", "ood_metrics.md")


def run(raw_csv, holdout_sector="52", epochs=20, seed=42, nrows=None):
    t0 = time.time()
    X, y = clean.load_clean(raw_csv, nrows=nrows)
    sector = X["naics_sector"].to_numpy().astype(object)
    ood_mask = sector == holdout_sector
    in_idx = np.where(~ood_mask)[0]
    ood_idx = np.where(ood_mask)[0]

    # split in-distribution data into train/test
    y_in = y.iloc[in_idx]
    itr, ite = split.stratified_split(y_in, test_size=0.2, seed=seed)
    tr, te = in_idx[itr], in_idx[ite]

    enc = encoders.TabularEncoder().fit(X.iloc[tr])
    xtr = enc.transform(X.iloc[tr]); ytr = y.iloc[tr].to_numpy()
    xte = enc.transform(X.iloc[te])
    xood = enc.transform(X.iloc[ood_idx])
    itr2, iva = split.stratified_split(y.iloc[tr], test_size=0.1, seed=seed)

    net = tabular_dl.build("mlp", enc.n_numeric, enc.cardinalities)
    net, _ = trainer.train(net, (xtr[0][itr2], xtr[1][itr2], ytr[itr2]),
                           (xtr[0][iva], xtr[1][iva], ytr[iva]), epochs=epochs, seed=seed, verbose=False)

    f_tr = uncertainty.extract_features(net, xtr[0], xtr[1])
    ood = uncertainty.MahalanobisOOD().fit(f_tr, ytr)
    s_in = ood.score(uncertainty.extract_features(net, xte[0], xte[1]))
    s_ood = ood.score(uncertainty.extract_features(net, xood[0], xood[1]))
    real_auroc = float(roc_auc_score(
        np.r_[np.zeros(len(s_in)), np.ones(len(s_ood))], np.r_[s_in, s_ood]))

    # synthetic permutation baseline (for context vs Phase C)
    rng = np.random.default_rng(seed)
    perm_n = np.column_stack([rng.permutation(xte[0][:, j]) for j in range(xte[0].shape[1])]).astype(np.float32)
    perm_c = np.column_stack([rng.permutation(xte[1][:, j]) for j in range(xte[1].shape[1])]).astype(np.int64)
    s_perm = ood.score(uncertainty.extract_features(net, perm_n, perm_c))
    perm_auroc = float(roc_auc_score(
        np.r_[np.zeros(len(s_in)), np.ones(len(s_perm))], np.r_[s_in, s_perm]))

    out = _write(holdout_sector, len(ood_idx), len(te), real_auroc, perm_auroc,
                 float(np.median(s_in)), float(np.median(s_ood)), time.time() - t0)
    print(f"[ood] wrote {out}  ({time.time()-t0:.1f}s)")
    return out


def _write(sector, n_ood, n_in, real_auroc, perm_auroc, med_in, med_ood, secs):
    L = ["# SBA Credit-Trust — E-4 Real OOD Benchmark (leave-one-sector-out)\n"]
    L.append(f"Held-out NAICS sector **{sector}** removed from training (n_OOD={n_ood:,}); Mahalanobis "
             f"detector fit on the remaining sectors. In-distribution test n={n_in:,}. Runtime {secs:.1f}s.\n")
    L.append("| OOD type | AUROC |")
    L.append("|---|---|")
    L.append(f"| **Real (unseen sector {sector})** | **{real_auroc:.4f}** |")
    L.append(f"| Synthetic (feature permutation, Phase-C style) | {perm_auroc:.4f} |")
    L.append(f"\n- Median Mahalanobis distance: in-distribution = {med_in:.1f}, unseen-sector = {med_ood:.1f}.")
    L.append("\n> A held-out industry is a genuine covariate shift; AUROC > 0.5 means the penultimate-"
             "layer Mahalanobis score flags it without ever being trained to. This replaces the "
             "synthetic-only OOD evidence flagged in AUDIT M-4.\n")
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
