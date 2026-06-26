---
title: SBA Credit-Trust
emoji: 🏦
colorFrom: indigo
colorTo: green
sdk: docker
app_port: 7860
pinned: false
license: mit
---

# SBA Credit-Trust — calibrated, selective loan-default decisioning

Interactive demo of a trustworthy credit-risk model on the public U.S. SBA loan dataset. For a
single loan it returns a **calibrated** default probability, a conformal **auto-approve /
auto-reject / refer-to-human** decision at a chosen risk target, and a **SHAP** explanation.

> Research / education only — not a real lending decision system. Probabilities are calibrated and
> the conformal risk control holds *in distribution*; it degrades under temporal shift (see the
> model card and the project's adaptive-conformal report).

## Deploy this Space

This folder is the deployable bundle. To publish:

```bash
# from the repo root, after `python -m src.serve.train_artifact`
cp -r src hf_space/src
cp models/serving_bundle.joblib hf_space/models/serving_bundle.joblib
# then push hf_space/ to a Hugging Face Space (SDK: streamlit)
```

The served model is the Optuna-tuned **XGBoost** + temperature calibration + conformal thresholds —
so the Space needs no GPU and no PyTorch.

Full project: leakage-corrected benchmark, tabular DL (MLP / FT-Transformer), calibration,
uncertainty, OOD, conformal selective prediction, cost-sensitive decisioning, temporal-shift study,
and adaptive conformal that restores risk control under shift.
