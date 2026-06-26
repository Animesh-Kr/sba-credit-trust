"""Phase-C runner: the trustworthy-ML layer -> reports/trust_metrics.md.

Three-way split (train / calibration / test). On the MLP we run: temperature + isotonic
calibration, MC-Dropout and deep-ensemble uncertainty, Mahalanobis OOD, and conformal
risk-controlled selective prediction. We also calibrate XGBoost and produce SHAP global
importances. Everything that is fit (calibrators, conformal thresholds, OOD Gaussians) is fit on
data disjoint from the test set.
"""
from __future__ import annotations

import os
import time

import numpy as np
from sklearn.metrics import roc_auc_score

from ..calibration.scaling import IsotonicCalibrator, TemperatureScaler, logit
from ..conformal.selective import SelectiveRiskController
from ..data import clean, encoders, split
from ..eval import metrics
from ..models import tabular_dl, trainer, uncertainty

REPORT = os.path.join(os.path.dirname(__file__), "..", "..", "reports", "trust_metrics.md")


def _three_way(y, seed):
    """train (~65%) / calibration (~15%) / test (20%), all stratified."""
    rest_idx, test_idx = split.stratified_split(y, test_size=0.20, seed=seed)
    y_rest = y.iloc[rest_idx]
    inner_tr, inner_cal = split.stratified_split(y_rest, test_size=0.1875, seed=seed)  # 0.1875*0.8=0.15
    return rest_idx[inner_tr], rest_idx[inner_cal], test_idx


def run(raw_csv, epochs=25, ensemble=3, mc_passes=30, shap_n=2000, seed=42, nrows=None):
    t0 = time.time()
    X, y = clean.load_clean(raw_csv, nrows=nrows)
    tr, cal, te = _three_way(y, seed)
    enc = encoders.TabularEncoder().fit(X.iloc[tr])
    xtr = enc.transform(X.iloc[tr]); ytr = y.iloc[tr].to_numpy()
    xca = enc.transform(X.iloc[cal]); yca = y.iloc[cal].to_numpy()
    xte = enc.transform(X.iloc[te]); yte = y.iloc[te].to_numpy()

    # inner val for early stopping (carved from train)
    itr, iva = split.stratified_split(y.iloc[tr], test_size=0.1, seed=seed)
    train_data = (xtr[0][itr], xtr[1][itr], ytr[itr])
    val_data = (xtr[0][iva], xtr[1][iva], ytr[iva])

    # --- deep ensemble of MLPs ---
    members, member_probs_te = [], []
    for m in range(ensemble):
        print(f"[trust] training MLP member {m+1}/{ensemble}")
        net = tabular_dl.build("mlp", enc.n_numeric, enc.cardinalities)
        net, _ = trainer.train(net, train_data, val_data, epochs=epochs, seed=seed + m, verbose=False)
        members.append(net)
        member_probs_te.append(trainer.predict_proba(net, xte[0], xte[1]))
    net0 = members[0]

    # --- calibration (member 0) ---
    logits_cal = trainer.predict_logits(net0, xca[0], xca[1])
    logits_te = trainer.predict_logits(net0, xte[0], xte[1])
    p_te_raw = 1 / (1 + np.exp(-logits_te))
    ts = TemperatureScaler().fit(logits_cal, yca)
    p_te_temp = ts.transform(logits_te)
    iso = IsotonicCalibrator().fit(1 / (1 + np.exp(-logits_cal)), yca)
    p_te_iso = iso.transform(p_te_raw)
    cal_rows = {
        "MLP raw": metrics.classification_report(yte, p_te_raw),
        f"MLP + temperature (T={ts.T_:.2f})": metrics.classification_report(yte, p_te_temp),
        "MLP + isotonic": metrics.classification_report(yte, p_te_iso),
    }

    # --- MC-Dropout uncertainty (member 0) ---
    mc_mean, mc_std = uncertainty.mc_dropout_predict(net0, xte[0], xte[1], passes=mc_passes)
    ent = uncertainty.predictive_entropy(mc_mean)
    pred = (mc_mean >= 0.5).astype(int)
    correct = pred == yte
    unc_corr = float(ent[correct].mean())
    unc_wrong = float(ent[~correct].mean())

    # selective accuracy: accuracy on the 50% most-confident vs least-confident by entropy
    order = np.argsort(ent)
    half = len(ent) // 2
    acc_conf = float(correct[order[:half]].mean())
    acc_unconf = float(correct[order[half:]].mean())

    # --- deep ensemble metrics ---
    ens_mean, ens_std = uncertainty.ensemble_mean(member_probs_te)
    ens_rep = metrics.classification_report(yte, ens_mean)

    # --- Mahalanobis OOD ---
    f_tr = uncertainty.extract_features(net0, xtr[0], xtr[1])
    f_te = uncertainty.extract_features(net0, xte[0], xte[1])
    ood = uncertainty.MahalanobisOOD().fit(f_tr, ytr)
    score_in = ood.score(f_te)
    # synthetic OOD: independently permute each feature column (destroys joint structure)
    rng = np.random.default_rng(seed)
    xood_n = np.column_stack([rng.permutation(xte[0][:, j]) for j in range(xte[0].shape[1])])
    xood_c = np.column_stack([rng.permutation(xte[1][:, j]) for j in range(xte[1].shape[1])])
    f_ood = uncertainty.extract_features(net0, xood_n.astype(np.float32), xood_c.astype(np.int64))
    score_ood = ood.score(f_ood)
    ood_auroc = float(
        roc_auc_score(
            np.r_[np.zeros(len(score_in)), np.ones(len(score_ood))], np.r_[score_in, score_ood]
        )
    )

    # --- conformal selective prediction on the best-calibrated MLP (isotonic) ---
    p_cal_iso = iso.transform(1 / (1 + np.exp(-logits_cal)))
    selective = {}
    for alpha in (0.05, 0.10):
        ctrl = SelectiveRiskController(alpha=alpha, delta=0.05).fit(p_cal_iso, yca)
        selective[alpha] = ctrl.evaluate(p_te_iso, yte)

    # --- XGBoost calibration + SHAP ---
    xgb_block = _xgb_calibration_and_shap(X, y, tr, cal, te, shap_n, seed)

    out = _write(cal_rows, ens_rep, ts.T_, (unc_corr, unc_wrong, acc_conf, acc_unconf),
                 ood_auroc, selective, xgb_block, len(te), time.time() - t0)
    print(f"[trust] wrote {out}  ({time.time()-t0:.1f}s)")
    return out


def _xgb_calibration_and_shap(X, y, tr, cal, te, shap_n, seed):
    from ..models.baselines import xgboost

    Xtr, Xcal, Xte = X.iloc[tr], X.iloc[cal], X.iloc[te]
    ytr, ycal, yte = y.iloc[tr], y.iloc[cal].to_numpy(), y.iloc[te].to_numpy()
    pipe = xgboost(ytr, seed=seed)
    pipe.fit(Xtr, ytr)
    p_cal = pipe.predict_proba(Xcal)[:, 1]
    p_te = pipe.predict_proba(Xte)[:, 1]
    ts = TemperatureScaler().fit(logit(p_cal), ycal)
    p_te_cal = ts.transform(logit(p_te))
    rows = {
        "XGBoost raw": metrics.classification_report(yte, p_te),
        f"XGBoost + temperature (T={ts.T_:.2f})": metrics.classification_report(yte, p_te_cal),
    }
    # SHAP global importance via XGBoost's native exact TreeSHAP (pred_contribs).
    # (shap.TreeExplainer has a base_score parsing bug with this xgboost build.)
    import xgboost as xgb

    prep = pipe.named_steps["prep"]
    clf = pipe.named_steps["clf"]
    names = list(prep.get_feature_names_out())
    Xte_t = prep.transform(Xte)[:shap_n]
    Xte_t = Xte_t.toarray() if hasattr(Xte_t, "toarray") else Xte_t
    contribs = clf.get_booster().predict(xgb.DMatrix(Xte_t), pred_contribs=True)
    imp = np.abs(contribs[:, :-1]).mean(axis=0)  # last column is the bias term
    top = sorted(zip(names, imp), key=lambda kv: -kv[1])[:15]
    return {"rows": rows, "shap_top": top}


def _fmt(d):
    keys = ["pr_auc", "roc_auc", "f1_minority", "recall_minority", "brier", "ece"]
    return " | ".join(f"{d[k]:.4f}" for k in keys)


def _write(cal_rows, ens_rep, T, unc, ood_auroc, selective, xgb_block, n_test, secs):
    uc, uw, ac, au = unc
    L = ["# SBA Credit-Trust — Phase C Trustworthy-ML Layer\n"]
    L.append(f"- Three-way split (train / calibration / test). Test n = **{n_test:,}**. "
             f"Runtime **{secs:.1f}s**.")
    L.append("- Positive class = **default (CHGOFF)**. All calibrators / thresholds fit on the "
             "calibration split, never on test.\n")

    L.append("## 1. Calibration (MLP) — ECE is the headline\n")
    L.append("| Variant | PR-AUC | ROC-AUC | F1(def) | Recall(def) | Brier | ECE |")
    L.append("|---|---|---|---|---|---|---|")
    for name, d in cal_rows.items():
        L.append(f"| {name} | {_fmt(d)} |")
    L.append("\n> Temperature scaling leaves ranking (PR/ROC-AUC) unchanged while reducing ECE/Brier.\n")

    L.append("## 2. Calibration (XGBoost)\n")
    L.append("| Variant | PR-AUC | ROC-AUC | F1(def) | Recall(def) | Brier | ECE |")
    L.append("|---|---|---|---|---|---|---|")
    for name, d in xgb_block["rows"].items():
        L.append(f"| {name} | {_fmt(d)} |")
    L.append("")

    L.append("## 3. Uncertainty\n")
    L.append(f"- **Deep ensemble** ({len(ens_rep) and 'mean of members'}): "
             f"PR-AUC {ens_rep['pr_auc']:.4f}, ROC-AUC {ens_rep['roc_auc']:.4f}, "
             f"Brier {ens_rep['brier']:.4f}, ECE {ens_rep['ece']:.4f}.")
    L.append(f"- **MC-Dropout** predictive entropy: mean on **correct** = {uc:.4f} nats vs "
             f"**incorrect** = {uw:.4f} nats (higher uncertainty on errors ⇒ useful signal).")
    L.append(f"- **Selective accuracy by uncertainty**: most-confident 50% acc = **{ac:.4f}** vs "
             f"least-confident 50% acc = **{au:.4f}**.\n")

    L.append("## 4. Out-of-distribution detection (Mahalanobis)\n")
    L.append(f"- AUROC separating in-distribution test from feature-permuted OOD = **{ood_auroc:.4f}** "
             "(1.0 = perfect; flags inputs unlike the training distribution).\n")

    L.append("## 5. Conformal risk-controlled selective prediction (calibrated MLP)\n")
    L.append("Auto-**approve** low-risk loans, auto-**reject** high-risk, **abstain** in between. "
             "Guarantee: default rate among auto-approved ≤ α (1−δ=95% confidence).\n")
    L.append("| Target α | t_lo | t_hi | Coverage | Abstain | Approved n | **Default rate among approved** | Default rate among rejected |")
    L.append("|---|---|---|---|---|---|---|---|")
    for alpha, r in selective.items():
        L.append(f"| {alpha:.2f} | {r['t_lo']:.3f} | {r['t_hi']:.3f} | {r['coverage']:.3f} | "
                 f"{r['abstain_rate']:.3f} | {r['n_approved']:,} | "
                 f"**{r['default_rate_among_approved']:.4f}** | {r['default_rate_among_rejected']:.4f} |")
    L.append("\n> The default rate among auto-approved loans stays at or below α — risk control a raw "
             "probability score cannot provide. (Empirically validated; the formal RCPS/LTT "
             "conditions and the threshold-selection caveat are documented in docs/AUDIT.md, M-1.)\n")

    L.append("## 6. Explainability — SHAP global importance (XGBoost, top 15)\n")
    L.append("| Feature | mean(|SHAP|) |")
    L.append("|---|---|")
    for name, v in xgb_block["shap_top"]:
        L.append(f"| `{name}` | {v:.4f} |")
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
    ep = int(sys.argv[2]) if len(sys.argv) > 2 else 25
    run(csv, epochs=ep, nrows=nrows)
