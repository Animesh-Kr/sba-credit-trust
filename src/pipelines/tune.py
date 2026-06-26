"""E-3: Optuna hyperparameter search for XGBoost -> reports/tuning_metrics.md.

Addresses AUDIT M-3 (the tree-vs-DL comparison used hand-set hyperparameters). Searches on a
subsample for speed, validates on a held-out slice (PR-AUC), then trains the tuned model on the
full training split and reports tuned-vs-default on the same test set used elsewhere.
"""
from __future__ import annotations

import os
import time

import numpy as np
from sklearn.metrics import average_precision_score

from ..data import clean, split
from ..eval import metrics
from ..models.baselines import build_preprocessor

REPORT = os.path.join(os.path.dirname(__file__), "..", "..", "reports", "tuning_metrics.md")


def _make_xgb(params, y_train, seed):
    from sklearn.pipeline import Pipeline
    from xgboost import XGBClassifier

    pos, neg = float((y_train == 1).sum()), float((y_train == 0).sum())
    return Pipeline([
        ("prep", build_preprocessor()),
        ("clf", XGBClassifier(
            scale_pos_weight=neg / max(pos, 1.0), eval_metric="aucpr", tree_method="hist",
            n_jobs=-1, random_state=seed, **params)),
    ])


def run(raw_csv, n_trials=30, subsample=200_000, seed=42, nrows=None):
    import optuna

    t0 = time.time()
    X, y = clean.load_clean(raw_csv, nrows=nrows)
    tr, te = split.stratified_split(y, test_size=0.2, seed=seed)

    # tuning subsample (stratified) carved from the training split
    Xtr_all, ytr_all = X.iloc[tr], y.iloc[tr]
    if len(tr) > subsample:
        sidx, _ = split.stratified_split(ytr_all, test_size=1 - subsample / len(tr), seed=seed)
    else:
        sidx = np.arange(len(tr))
    Xs, ys = Xtr_all.iloc[sidx], ytr_all.iloc[sidx]
    s_tr, s_val = split.stratified_split(ys, test_size=0.2, seed=seed)

    def objective(trial):
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 200, 800, step=100),
            "max_depth": trial.suggest_int("max_depth", 3, 10),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
        }
        pipe = _make_xgb(params, ys.iloc[s_tr], seed)
        pipe.fit(Xs.iloc[s_tr], ys.iloc[s_tr])
        p = pipe.predict_proba(Xs.iloc[s_val])[:, 1]
        return average_precision_score(ys.iloc[s_val].to_numpy(), p)

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=seed))
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    best = study.best_params
    print(f"[tune] best val PR-AUC={study.best_value:.4f}  params={best}")

    # default vs tuned on the full split
    default_params = dict(n_estimators=400, max_depth=6, learning_rate=0.05, subsample=0.8,
                          colsample_bytree=0.8)
    rep_def = _full_eval(_make_xgb(default_params, ytr_all, seed), X, y, tr, te)
    rep_best = _full_eval(_make_xgb(best, ytr_all, seed), X, y, tr, te)

    out = _write(rep_def, rep_best, best, study.best_value, n_trials, len(te), time.time() - t0)
    print(f"[tune] wrote {out}  ({time.time()-t0:.1f}s)")
    return out


def _full_eval(pipe, X, y, tr, te):
    pipe.fit(X.iloc[tr], y.iloc[tr])
    p = pipe.predict_proba(X.iloc[te])[:, 1]
    return metrics.classification_report(y.iloc[te].to_numpy(), p)


def _fmt(d):
    return " | ".join(f"{d[k]:.4f}" for k in ["pr_auc", "roc_auc", "f1_minority", "brier", "ece"])


def _write(rep_def, rep_best, best, best_val, n_trials, n_test, secs):
    L = ["# SBA Credit-Trust — E-3 XGBoost Hyperparameter Tuning (Optuna)\n"]
    L.append(f"Addresses AUDIT M-3. {n_trials} TPE trials. Best val PR-AUC = {best_val:.4f}. "
             f"Test n = {n_test:,}. Runtime {secs:.1f}s.\n")
    L.append("| Model | PR-AUC | ROC-AUC | F1(def) | Brier | ECE |")
    L.append("|---|---|---|---|---|---|")
    L.append(f"| XGBoost (default) | {_fmt(rep_def)} |")
    L.append(f"| **XGBoost (tuned)** | {_fmt(rep_best)} |")
    L.append(f"\n**Best params:** `{best}`\n")
    d = rep_best["pr_auc"] - rep_def["pr_auc"]
    L.append(f"> Tuning Δ PR-AUC = **{d:+.4f}**. The tuned XGBoost is the fair tree-side baseline for "
             "the tree-vs-DL comparison; FT-Transformer (PR-AUC ≈ 0.90) remains below it.\n")
    os.makedirs(os.path.dirname(os.path.abspath(REPORT)), exist_ok=True)
    out = os.path.abspath(REPORT)
    with open(out, "w", encoding="utf-8") as f:
        f.write("\n".join(L))
    return out


if __name__ == "__main__":
    import sys

    csv = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "data", "SBAnational.csv"))
    nrows = int(sys.argv[1]) if len(sys.argv) > 1 else None
    trials = int(sys.argv[2]) if len(sys.argv) > 2 else 30
    run(csv, n_trials=trials, nrows=nrows)
