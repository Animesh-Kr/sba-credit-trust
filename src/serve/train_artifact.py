"""Train and persist the production serving bundle.

Serves the *tuned* XGBoost (best PR-AUC, fast, ONNX-friendly) with temperature calibration and
conformal approve/reject thresholds. Run: ``python -m src.serve.train_artifact``.
"""
from __future__ import annotations

import time

import numpy as np
from sklearn.pipeline import Pipeline

from ..calibration.scaling import TemperatureScaler, logit
from ..conformal.selective import SelectiveRiskController
from ..data import clean, schema, split
from ..eval import metrics
from ..models.baselines import build_preprocessor
from .bundle import DEFAULT_PATH, ServingBundle

# best hyperparameters from the Optuna search (reports/tuning_metrics.md, E-3)
TUNED = dict(n_estimators=700, max_depth=8, learning_rate=0.05116500998797686,
             subsample=0.8249502727149073, colsample_bytree=0.8124819686258051,
             min_child_weight=4, reg_lambda=0.054683267586723545)


def _build_choices(X, cols, min_count=200):
    choices = {}
    for c in cols:
        vc = X[c].astype("object").value_counts()
        choices[c] = sorted(str(v) for v in vc[vc >= min_count].index)
    return choices


def run(raw_csv, seed=42, nrows=None, path=DEFAULT_PATH):
    from xgboost import XGBClassifier

    t0 = time.time()
    X, y = clean.load_clean(raw_csv, nrows=nrows)
    rest, te = split.stratified_split(y, test_size=0.2, seed=seed)
    itr, ical = split.stratified_split(y.iloc[rest], test_size=0.15, seed=seed)
    tr, cal = rest[itr], rest[ical]

    pos, neg = float((y.iloc[tr] == 1).sum()), float((y.iloc[tr] == 0).sum())
    pipe = Pipeline([
        ("prep", build_preprocessor()),
        ("clf", XGBClassifier(scale_pos_weight=neg / max(pos, 1.0), eval_metric="aucpr",
                              tree_method="hist", n_jobs=-1, random_state=seed, **TUNED)),
    ])
    pipe.fit(X.iloc[tr], y.iloc[tr])

    # temperature calibration on the calibration slice
    p_cal_raw = pipe.predict_proba(X.iloc[cal])[:, 1]
    ts = TemperatureScaler().fit(logit(p_cal_raw), y.iloc[cal].to_numpy())

    # conformal thresholds (apply temperature first) for two risk targets
    p_cal = ts.transform(logit(p_cal_raw))
    conformal = {}
    for alpha in (0.05, 0.10):
        c = SelectiveRiskController(alpha=alpha, delta=0.05).fit(p_cal, y.iloc[cal].to_numpy())
        conformal[alpha] = {"t_lo": c.t_lo, "t_hi": c.t_hi}

    # held-out test metrics for the model card / app footer
    p_te = ts.transform(logit(pipe.predict_proba(X.iloc[te])[:, 1]))
    rep = metrics.classification_report(y.iloc[te].to_numpy(), p_te)

    numeric = schema.NUMERIC_FEATURES + schema.ENGINEERED_NUMERIC
    categorical = schema.CATEGORICAL_FEATURES
    bundle = ServingBundle(
        pipeline=pipe,
        temperature=ts.T_,
        conformal=conformal,
        numeric_cols=numeric,
        categorical_cols=categorical,
        choices=_build_choices(X.iloc[tr], categorical),
        numeric_defaults={c: float(np.nanmedian(X[c].astype(float))) for c in numeric},
        metadata={
            "model": "XGBoost (Optuna-tuned) + temperature calibration + conformal selection",
            "test_pr_auc": rep["pr_auc"], "test_roc_auc": rep["roc_auc"],
            "test_ece": rep["ece"], "test_brier": rep["brier"],
            "prevalence": float(y.mean()), "temperature": ts.T_,
            "n_train": int(len(tr)), "n_test": int(len(te)),
            "disclaimer": "Research/education only. Not a real lending decision system.",
        },
    )
    out = bundle.save(path)
    print(f"[artifact] saved {out}  ({time.time()-t0:.1f}s)  "
          f"PR-AUC={rep['pr_auc']:.4f} ECE={rep['ece']:.4f} T={ts.T_:.3f}")
    print(f"[artifact] conformal: {conformal}")
    return out


if __name__ == "__main__":
    import os
    import sys

    csv = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "data", "SBAnational.csv"))
    nrows = int(sys.argv[1]) if len(sys.argv) > 1 else None
    run(csv, nrows=nrows)
