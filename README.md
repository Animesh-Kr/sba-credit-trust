# SBA Credit-Trust: Calibrated, Selective Deep Learning for Small-Business Loan Decisions

> Trustworthy deep learning for U.S. SBA loan-default prediction — **calibrated probabilities,
> quantified uncertainty, conformal selective prediction, and cost-sensitive decisioning** on a
> widely-mis-modelled public dataset, with a **leakage-corrected** benchmark.

**Author:** Animesh Kumar · MSc Advanced Computer Science, Newcastle University

---

## Why this project

On tabular data, well-tuned tree ensembles often match or beat deep nets
([Grinsztajn et al., 2022](https://arxiv.org/abs/2207.08815)). So the contribution here is **not**
"DL beats GBM on accuracy." It is **trustworthiness**: calibration (ECE/Brier), uncertainty
(MC-Dropout / deep ensembles), out-of-distribution detection (Mahalanobis), **conformal
risk-controlled selective prediction** (bounded default rate among auto-approved loans, with an
abstention option), and **cost-sensitive, dollar-aware decisions** — value a raw probability score
does not provide.

A second, standalone contribution: a **leakage-free benchmark**. The widely-circulated public demo
on this dataset oversamples *before* the train/test split, which inflates its reported F1. This repo
re-establishes an honest baseline (resampling inside CV folds only) and reports PR-AUC, ROC-AUC,
minority-class F1, Brier, and ECE.

> **Integrity:** The dataset is public. The reference repo (Bagja Satiaraharja) was used only to
> understand features and prior work — **no code, models, or metrics are copied**. Every number in
> this repo comes from runs in this repo.

## Dataset

`SBAnational.csv` (~899k loans) — Kaggle:
[`mirbektoktogaraev/should-this-loan-be-approved-or-denied`](https://www.kaggle.com/datasets/mirbektoktogaraev/should-this-loan-be-approved-or-denied).
Pulled via `python -m src.data.download` (not committed). Target: `MIS_Status` →
**default (CHGOFF) = positive/event class**, ~17–18% prevalence.

## Quickstart

```bash
pip install -r requirements.txt
python -m src.data.download          # pull SBAnational.csv from Kaggle into data/
pytest -q                            # run the test suite
python -m src.pipelines.eda          # leakage audit + EDA -> reports/
python -m src.pipelines.baseline     # leakage-free LR + XGBoost baseline -> reports/baseline_metrics.md
```

## Layout

```
src/
  data/        download · schema (target + leakage list) · clean · split (stratified + temporal)
  eval/        metrics (pr-auc, roc-auc, f1-min, brier, ece)
  models/      leakage-free baselines (LR, XGBoost), CV-internal resampling
  calibration/ (Phase C) temperature / isotonic scaling
  conformal/   (Phase C) split-conformal selective prediction
  pipelines/   eda · baseline · (later) deep models, trust layer
  app/         (Phase E) Streamlit
configs/       data.yaml · baseline.yaml
tests/         pytest — written before implementation (TDD)
```

## Roadmap

- **A — Foundations** ✅ leakage-safe pipeline, EDA, leakage-corrected XGBoost baseline (`baseline_metrics.md`).
- **B — Tabular DL** ✅ MLP+embeddings, FT-Transformer, focal loss (`deep_metrics.md`).
- **C — Trust layer** ✅ calibration, MC-Dropout/ensembles, Mahalanobis OOD, conformal, SHAP (`trust_metrics.md`).
- **D — Novelty** ✅ cost-sensitive policy, temporal-shift study, slice analysis (`novelty_metrics.md`).
- **Audit + enhancements** ✅ adversarial audit (`docs/AUDIT.md`); repeated-seed CIs (`robustness_metrics.md`),
  rigorous + **adaptive conformal that restores risk control under shift** (`adaptive_conformal_metrics.md`),
  Optuna tuning (`tuning_metrics.md`), fairness disparity (`fairness_metrics.md`), best-model conformal
  (`best_model_conformal.md`), real OOD (`ood_metrics.md`), dataset checksum + determinism + CI.
- **E — Deploy:** Streamlit + ONNX + HF Space + model card. ⬅ next
- **F — Writeup:** short honest preprint (Zenodo DOI).

### Reproduce the analyses

```bash
python -m src.pipelines.eda          # leakage audit + EDA
python -m src.pipelines.baseline     # leakage-free LR + XGBoost
python -m src.pipelines.deep         # MLP + FT-Transformer
python -m src.pipelines.trust        # calibration, uncertainty, OOD, conformal, SHAP
python -m src.pipelines.novelty      # temporal shift, cost-sensitive policy, slices
python -m src.pipelines.robustness   # repeated-seed CIs (E-2)
python -m src.pipelines.adaptive_conformal  # rigorous + adaptive conformal (E-1/E-9)
python -m src.pipelines.tune         # Optuna XGBoost tuning (E-3)
python -m src.pipelines.fairness     # disparity audit (E-10)
python -m src.pipelines.ood          # leave-one-sector OOD (E-4)
python -m src.pipelines.best_model_conformal  # FT-Transformer + conformal (E-8)
```

### Deploy (Phase E)

```bash
python -m src.serve.train_artifact   # train + persist the serving bundle (tuned XGBoost + calib + conformal)
python -m src.serve.export_onnx      # ONNX export + parity + latency benchmark
streamlit run src/app/streamlit_app.py   # interactive app: prob + approve/reject/refer + SHAP
```

The Hugging Face Space scaffold is in `hf_space/` (streamlit SDK; XGBoost serving, no GPU).

## License

Code: MIT. Dataset under its original Kaggle terms.
