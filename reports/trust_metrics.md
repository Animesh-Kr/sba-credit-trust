# SBA Credit-Trust — Phase C Trustworthy-ML Layer

- Three-way split (train / calibration / test). Test n = **179,434**. Runtime **813.6s**.
- Positive class = **default (CHGOFF)**. All calibrators / thresholds fit on the calibration split, never on test.

## 1. Calibration (MLP) — ECE is the headline

| Variant | PR-AUC | ROC-AUC | F1(def) | Recall(def) | Brier | ECE |
|---|---|---|---|---|---|---|
| MLP raw | 0.8567 | 0.9531 | 0.7825 | 0.8215 | 0.0985 | 0.1932 |
| MLP + temperature (T=0.31) | 0.8567 | 0.9531 | 0.7825 | 0.8215 | 0.0619 | 0.0519 |
| MLP + isotonic | 0.8522 | 0.9530 | 0.7869 | 0.7393 | 0.0544 | 0.0031 |

> Temperature scaling leaves ranking (PR/ROC-AUC) unchanged while reducing ECE/Brier.

## 2. Calibration (XGBoost)

| Variant | PR-AUC | ROC-AUC | F1(def) | Recall(def) | Brier | ECE |
|---|---|---|---|---|---|---|
| XGBoost raw | 0.9126 | 0.9781 | 0.8193 | 0.9330 | 0.0538 | 0.0744 |
| XGBoost + temperature (T=0.89) | 0.9126 | 0.9781 | 0.8193 | 0.9330 | 0.0535 | 0.0691 |

## 3. Uncertainty

- **Deep ensemble** (mean of members): PR-AUC 0.8609, ROC-AUC 0.9553, Brier 0.0968, ECE 0.1923.
- **MC-Dropout** predictive entropy: mean on **correct** = 0.5268 nats vs **incorrect** = 0.6572 nats (higher uncertainty on errors ⇒ useful signal).
- **Selective accuracy by uncertainty**: most-confident 50% acc = **0.9875** vs least-confident 50% acc = **0.8481**.

## 4. Out-of-distribution detection (Mahalanobis)

- AUROC separating in-distribution test from feature-permuted OOD = **0.7956** (1.0 = perfect; flags inputs unlike the training distribution).

## 5. Conformal risk-controlled selective prediction (calibrated MLP)

Auto-**approve** low-risk loans, auto-**reject** high-risk, **abstain** in between. Guarantee: default rate among auto-approved ≤ α (1−δ=95% confidence).

| Target α | t_lo | t_hi | Coverage | Abstain | Approved n | **Default rate among approved** | Default rate among rejected |
|---|---|---|---|---|---|---|---|
| 0.05 | 0.394 | 0.931 | 0.906 | 0.094 | 150,091 | **0.0499** | 0.9545 |
| 0.10 | 0.879 | 0.742 | 0.967 | 0.033 | 157,148 | **0.0729** | 0.9384 |

> The default rate among auto-approved loans stays at or below α — risk control a raw probability score cannot provide. (Empirically validated; the formal RCPS/LTT conditions and the threshold-selection caveat are documented in docs/AUDIT.md, M-1.)

## 6. Explainability — SHAP global importance (XGBoost, top 15)

| Feature | mean(|SHAP|) |
|---|---|
| `num__Term` | 2.2184 |
| `num__ApprovalFY` | 0.6191 |
| `num__sba_portion` | 0.4746 |
| `num__SBA_Appv` | 0.4680 |
| `cat__BankState_HI` | 0.4373 |
| `cat__BankState_MN` | 0.3865 |
| `cat__BankState_VT` | 0.3549 |
| `cat__BankState_MT` | 0.3432 |
| `cat__BankState_IN` | 0.3420 |
| `cat__BankState_WY` | 0.3353 |
| `cat__State_GA` | 0.2965 |
| `cat__State_NV` | 0.2826 |
| `cat__BankState_PA` | 0.2819 |
| `num__GrAppv` | 0.2802 |
| `cat__BankState_GA` | 0.2644 |
