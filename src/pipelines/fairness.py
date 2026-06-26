"""E-10 runner -> reports/fairness_metrics.md. Disparity audit across state, loan-size, sector."""
from __future__ import annotations

import os
import time

import numpy as np

from ..calibration.scaling import TemperatureScaler, logit
from ..data import clean, split
from ..eval import fairness
from ..models.baselines import xgboost

REPORT = os.path.join(os.path.dirname(__file__), "..", "..", "reports", "fairness_metrics.md")


def run(raw_csv, seed=42, nrows=None):
    t0 = time.time()
    X, y, meta = clean.load_with_meta(raw_csv, nrows=nrows)
    tr, te = split.stratified_split(y, test_size=0.2, seed=seed)
    itr, ical = split.stratified_split(y.iloc[tr], test_size=0.15, seed=seed)
    pipe = xgboost(y.iloc[tr[itr]], seed=seed)
    pipe.fit(X.iloc[tr[itr]], y.iloc[tr[itr]])
    ts = TemperatureScaler().fit(logit(pipe.predict_proba(X.iloc[tr[ical]])[:, 1]), y.iloc[tr[ical]].to_numpy())
    p_te = ts.transform(logit(pipe.predict_proba(X.iloc[te])[:, 1]))
    y_te = y.iloc[te].to_numpy()

    gr = meta["GrAppv"].to_numpy()[te]
    q = np.nanquantile(gr, [0.25, 0.5, 0.75])
    size_band = np.digitize(gr, q)
    size_labels = {0: f"Q1<=${q[0]:,.0f}", 1: "Q2", 2: "Q3", 3: f"Q4>${q[2]:,.0f}"}

    groups = {
        "State": X["State"].to_numpy()[te],
        "Loan-size band": np.array([size_labels[b] for b in size_band], dtype=object),
        "NAICS sector": meta["naics_sector"].to_numpy()[te],
    }
    blocks = {}
    for name, g in groups.items():
        rows = fairness.group_metrics(y_te, p_te, g, threshold=0.5, min_n=2000)
        blocks[name] = (rows, fairness.disparity(rows))

    out = _write(blocks, time.time() - t0)
    print(f"[fairness] wrote {out}  ({time.time()-t0:.1f}s)")
    return out


def _write(blocks, secs):
    L = ["# SBA Credit-Trust — E-10 Fairness / Disparity Audit\n"]
    L.append(f"Temperature-calibrated XGBoost, random-split test. Approve iff P(default) < 0.5. "
             f"Runtime {secs:.1f}s.\n")
    L.append("Favourable outcome = approved. `*_gap` = max−min across groups (0 = perfectly fair).\n")
    for name, (rows, disp) in blocks.items():
        L.append(f"## By {name}\n")
        L.append("| Group | n | base default | approval rate | approval rate (good) | default-catch | ECE |")
        L.append("|---|---|---|---|---|---|---|")
        for g, r in sorted(rows.items(), key=lambda kv: -kv[1]["n"])[:12]:
            L.append(f"| {g} | {r['n']:,} | {r['base_default_rate']:.3f} | {r['approval_rate']:.3f} | "
                     f"{r['approval_rate_good']:.3f} | {r['default_catch_rate']:.3f} | {r['ece']:.3f} |")
        L.append(f"\n**Disparity gaps** — approval {disp['approval_rate_gap']:.3f}, "
                 f"equal-opportunity (approval|good) {disp['approval_rate_good_gap']:.3f}, "
                 f"default-catch {disp['default_catch_rate_gap']:.3f}, ECE {disp['ece_gap']:.3f}.\n")
    L.append("> Disparity gaps quantify how unevenly the decision and its calibration fall across "
             "groups. Geographic/industry gaps would require mitigation before any real deployment.\n")
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
