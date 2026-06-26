# SBA Credit-Trust — Phase B Tabular Deep Learning

- Samples: **897,167**  ·  features: **17**  ·  default prevalence: **0.1756**
- Split: same stratified hold-out as Phase A — test **179,434**
- include_disbursement: **False**  ·  runtime: **1097.3s**

Held-out test metrics. Positive class = **default (CHGOFF)**. Threshold = 0.5. Focal loss (α=0.75, γ=2), early stop on val PR-AUC.

| Model | PR-AUC | ROC-AUC | F1(def) | Prec(def) | Recall(def) | Brier | ECE |
|---|---|---|---|---|---|---|---|
| _XGBoost (Phase A ref)_ | 0.9139 | 0.9785 | 0.8200 | 0.7316 | 0.9328 | 0.0533 | 0.0741 |
| MLP + embeddings | 0.8672 | 0.9583 | 0.7913 | 0.7505 | 0.8368 | 0.0931 | 0.1846 |
| FT-Transformer | 0.9014 | 0.9730 | 0.8183 | 0.7661 | 0.8781 | 0.0697 | 0.1447 |

> Per the project thesis, DL is **not** expected to beat XGBoost on accuracy on tabular data (Grinsztajn et al. 2022). These DL models are the substrate for the Phase-C trust layer (calibration, MC-Dropout uncertainty, Mahalanobis OOD, conformal selective prediction) — the actual contribution.
