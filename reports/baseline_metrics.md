# SBA Credit-Trust — Phase A Baseline (leakage-free)

- Samples: **897,167**  ·  features: **17**  ·  default prevalence: **0.1756**
- Split: stratified hold-out — train **717,733** / test **179,434**
- include_disbursement: **False**  ·  runtime: **72.3s**

All metrics are on the held-out test set. Positive class = **default (CHGOFF)**. Threshold = 0.5.

| Model | PR-AUC | ROC-AUC | F1(def) | Prec(def) | Recall(def) | Brier | ECE |
|---|---|---|---|---|---|---|---|
| LogisticRegression | 0.5632 | 0.8456 | 0.5337 | 0.3952 | 0.8214 | 0.1665 | 0.2160 |
| XGBoost | 0.9139 | 0.9785 | 0.8200 | 0.7316 | 0.9328 | 0.0533 | 0.0741 |

## Leakage demonstration (do NOT trust these numbers)

Reproducing the reference demo's flaw — random oversampling of the minority class **before** the train/test split, so duplicated default rows appear in both. The inflated test metrics below are the over-reporting artefact our leakage-free pipeline corrects:

| Pipeline | PR-AUC | ROC-AUC | F1(def) | Prec(def) | Recall(def) | Brier | ECE |
|---|---|---|---|---|---|---|---|
| XGBoost (oversample-before-split, LEAKY) | 0.9778 | 0.9799 | 0.9329 | 0.9292 | 0.9367 | 0.0509 | 0.0172 |

> Honest XGBoost PR-AUC = **0.9139**; leaky PR-AUC = **0.9778** → over-report of **+0.0638** purely from pipeline order.
