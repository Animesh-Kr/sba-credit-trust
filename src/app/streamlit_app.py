"""Streamlit app: a trustworthy SBA loan-default decision tool.

For a single loan it shows the **calibrated** default probability, a conformal **auto-approve /
auto-reject / abstain (refer)** decision at a chosen risk target, and a **SHAP** explanation of the
drivers. This is the deployable face of the project — calibration + selective risk + explainability,
not a bare yes/no. Research/education only.
"""
from __future__ import annotations

import os

import numpy as np
import pandas as pd
import streamlit as st

from src.serve.bundle import DEFAULT_PATH, ServingBundle

st.set_page_config(page_title="SBA Credit-Trust", page_icon="🏦", layout="centered")


@st.cache_resource
def _load(path: str) -> ServingBundle:
    return ServingBundle.load(path)


def _shap_local(bundle, X_row: pd.DataFrame, top_k: int = 6):
    """Per-feature contributions for this loan via XGBoost's exact TreeSHAP."""
    import xgboost as xgb

    prep = bundle.pipeline.named_steps["prep"]
    clf = bundle.pipeline.named_steps["clf"]
    names = list(prep.get_feature_names_out())
    Xt = prep.transform(X_row)
    Xt = Xt.toarray() if hasattr(Xt, "toarray") else Xt
    contribs = clf.get_booster().predict(xgb.DMatrix(Xt), pred_contribs=True)[0][:-1]
    pairs = sorted(zip(names, contribs), key=lambda kv: -abs(kv[1]))[:top_k]
    return pairs


def main():
    bundle_path = os.environ.get("SBA_BUNDLE", DEFAULT_PATH)
    if not os.path.exists(bundle_path):
        st.error(f"Model bundle not found at {bundle_path}. Run `python -m src.serve.train_artifact`.")
        return
    b = _load(bundle_path)
    m = b.metadata

    st.title("🏦 SBA Credit-Trust")
    st.caption("Calibrated, selective deep-credit-risk decisioning — probability + uncertainty-aware "
               "approve / reject / **refer** decision + explanation. " + m.get("disclaimer", ""))

    with st.sidebar:
        st.header("Risk target")
        alpha = st.selectbox("Max default rate among auto-approved (α)", list(b.conformal.keys()),
                             format_func=lambda a: f"{a:.0%}")
        st.markdown("---")
        st.metric("Model PR-AUC (test)", f"{m.get('test_pr_auc', float('nan')):.3f}")
        st.metric("Calibration ECE (test)", f"{m.get('test_ece', float('nan')):.3f}")
        st.caption(f"{m.get('model','')}\n\nBase default rate ≈ {m.get('prevalence',0):.1%}.")

    st.subheader("Loan application")
    c1, c2, c3 = st.columns(3)
    with c1:
        gr = st.number_input("Gross approved ($)", 1_000, 5_000_000, 150_000, step=5_000)
        term = st.number_input("Term (months)", 1, 480, 84)
        noemp = st.number_input("# Employees", 0, 10_000, 5)
    with c2:
        sba = st.number_input("SBA guaranteed ($)", 0, 5_000_000, 100_000, step=5_000)
        fy = st.number_input("Approval year", 1990, 2025, 2005)
        create = st.number_input("Jobs created", 0, 10_000, 0)
    with c3:
        retain = st.number_input("Jobs retained", 0, 10_000, 0)
        new_exist = st.selectbox("Business", ["1 (existing)", "2 (new)"])
        franchise = st.selectbox("Franchise?", ["No", "Yes"])

    c4, c5, c6 = st.columns(3)
    with c4:
        state = st.selectbox("Borrower state", b.choices.get("State", ["CA"]))
        revline = st.selectbox("Revolving line of credit", ["N", "Y"])
    with c5:
        bankstate = st.selectbox("Bank state", b.choices.get("BankState", ["CA"]))
        lowdoc = st.selectbox("LowDoc program", ["N", "Y"])
    with c6:
        sector = st.selectbox("NAICS sector (2-digit)", b.choices.get("naics_sector", ["44"]))
        urban = st.selectbox("Urban / rural", ["0 (unknown)", "1 (urban)", "2 (rural)"])

    if st.button("Assess loan", type="primary"):
        row = {
            "Term": term, "NoEmp": noemp, "CreateJob": create, "RetainedJob": retain,
            "GrAppv": gr, "SBA_Appv": sba, "ApprovalFY": fy,
            "sba_portion": (sba / gr) if gr else np.nan, "real_estate": int(term >= 240),
            "State": state, "BankState": bankstate, "NewExist": new_exist.split()[0],
            "UrbanRural": urban.split()[0], "RevLineCr": revline, "LowDoc": lowdoc,
            "naics_sector": str(sector), "is_franchise": int(franchise == "Yes"),
        }
        X = pd.DataFrame([row])[b.numeric_cols + b.categorical_cols]
        prob = float(b.predict_proba(X)[0])
        decision = b.decide(np.array([prob]), alpha=alpha)[0]

        st.markdown("---")
        cols = st.columns(2)
        cols[0].metric("Calibrated P(default)", f"{prob:.1%}")
        color = {"auto-approve": "🟢", "auto-reject": "🔴", "abstain / refer": "🟡"}[decision]
        cols[1].metric("Decision", f"{color} {decision}")
        if decision == "abstain / refer":
            st.info("The model is not confident enough to auto-decide at this risk target — refer to "
                    "a human underwriter.")
        st.caption(f"At α={alpha:.0%}, auto-approved loans are calibrated to default at ≤ {alpha:.0%} "
                   "(in-distribution; see temporal-shift caveat in the model card).")

        st.subheader("Why — top drivers (SHAP)")
        for name, val in _shap_local(b, X):
            arrow = "↑ risk" if val > 0 else "↓ risk"
            st.write(f"- `{name.replace('num__','').replace('cat__','')}` — {arrow} ({val:+.2f} logit)")


if __name__ == "__main__":
    main()
