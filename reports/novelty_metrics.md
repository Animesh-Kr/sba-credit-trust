# SBA Credit-Trust — Phase D Novelty Experiments

Runtime **65.1s**. Workhorse = temperature-calibrated XGBoost.

## A. Temporal distribution shift (the honest deployment number)

Train on cohorts **≤ 2006**, test on **> 2006** (n=168,346, spans the 2008 crisis).

| Evaluation | PR-AUC | ROC-AUC | F1(def) | Brier | ECE |
|---|---|---|---|---|---|
| Random split (optimistic) | 0.9137 | 0.9784 | 0.8215 | 0.0530 | 0.0685 |
| **Temporal split (deployment)** | 0.9161 | 0.9600 | 0.8223 | 0.0968 | 0.1052 |

Per-year on the future side:

| ApprovalFY | n | PR-AUC | ROC-AUC | ECE | actual default rate |
|---|---|---|---|---|---|
| 2007 | 71,649 | 0.949 | 0.970 | 0.073 | 0.428 |
| 2008 | 39,458 | 0.936 | 0.960 | 0.092 | 0.412 |
| 2009 | 19,103 | 0.735 | 0.911 | 0.125 | 0.208 |
| 2010 | 16,828 | 0.563 | 0.884 | 0.130 | 0.137 |
| 2011 | 12,593 | 0.595 | 0.926 | 0.189 | 0.079 |
| 2012 | 5,992 | 0.551 | 0.951 | 0.210 | 0.057 |
| 2013 | 2,455 | 0.219 | 0.929 | 0.223 | 0.029 |

**Conformal coverage under shift** (threshold fit in-era, α=0.05):

| Applied to | Coverage | Default among approved | Guarantee (≤0.05)? |
|---|---|---|---|
| In-era calibration | 0.929 | 0.0494 | ✅ |
| **Future cohorts (shifted)** | 0.817 | 0.1067 | ❌ broken by shift |

> A guarantee calibrated in one era can **break under distribution shift** — an honest, important caveat for deploying conformal methods on non-exchangeable temporal data.

## B. Cost-sensitive dollar-aware decisioning (random-split test)

Empirical loss-given-default estimated on train: **LGD = 64.06%** of gross principal. The EV rule approves iff `(1−p)·m·GrAppv > p·LGD·GrAppv`, i.e. a principled probability cut-off `t* = m/(m+LGD)` — contrast the arbitrary 0.5. Defaults costed at actual `ChgOffPrinGr` (ex-post accounting only).


**Margin m = 3%**  (EV-optimal cut-off t* = 0.045)

| Policy | Net $ | Approved | Defaults approved | Approval rate |
|---|---|---|---|---|
| approve_all | $-1,497,874,620 | 179,434 | 31,512 | 1.000 |
| threshold_0.5 | $593,770,935 | 139,317 | 2,092 | 0.776 |
| dollar_aware_EV | $596,144,039 | 100,957 | 279 | 0.563 |

> Dollar-aware vs 0.5-threshold: **$+2,373,104** net, and **+1,813** fewer defaulters approved — an explicit risk/return trade-off the accuracy threshold cannot express.


**Margin m = 5%**  (EV-optimal cut-off t* = 0.072)

| Policy | Net $ | Approved | Defaults approved | Approval rate |
|---|---|---|---|---|
| approve_all | $-879,633,096 | 179,434 | 31,512 | 1.000 |
| threshold_0.5 | $1,166,062,817 | 139,317 | 2,092 | 0.776 |
| dollar_aware_EV | $1,090,056,463 | 110,029 | 411 | 0.613 |

> Dollar-aware vs 0.5-threshold: **$-76,006,354** net, and **+1,681** fewer defaulters approved — an explicit risk/return trade-off the accuracy threshold cannot express.


**Margin m = 10%**  (EV-optimal cut-off t* = 0.135)

| Policy | Net $ | Approved | Defaults approved | Approval rate |
|---|---|---|---|---|
| approve_all | $665,970,714 | 179,434 | 31,512 | 1.000 |
| threshold_0.5 | $2,596,792,523 | 139,317 | 2,092 | 0.776 |
| dollar_aware_EV | $2,393,900,506 | 119,294 | 608 | 0.665 |

> Dollar-aware vs 0.5-threshold: **$-202,892,017** net, and **+1,484** fewer defaulters approved — an explicit risk/return trade-off the accuracy threshold cannot express.

## C. Slice / fairness audit (random-split test)

Performance + calibration must hold across groups, not just on average.

### By NAICS sector (top 10 by volume)

| Sector | n | PR-AUC | ROC-AUC | ECE | default rate |
|---|---|---|---|---|---|
| 44 | 16,899 | 0.923 | 0.978 | 0.077 | 0.227 |
| 81 | 14,380 | 0.920 | 0.979 | 0.067 | 0.199 |
| 54 | 13,613 | 0.924 | 0.982 | 0.061 | 0.188 |
| 23 | 13,404 | 0.935 | 0.979 | 0.073 | 0.233 |
| 72 | 13,397 | 0.895 | 0.970 | 0.089 | 0.218 |
| 62 | 11,009 | 0.874 | 0.976 | 0.058 | 0.105 |
| 42 | 9,717 | 0.940 | 0.983 | 0.070 | 0.197 |
| 45 | 8,601 | 0.923 | 0.975 | 0.072 | 0.231 |
| 33 | 7,524 | 0.884 | 0.974 | 0.075 | 0.137 |
| 56 | 6,518 | 0.928 | 0.979 | 0.076 | 0.238 |

### By loan-size band (GrAppv quartiles)

| Band | n | PR-AUC | ROC-AUC | ECE | default rate |
|---|---|---|---|---|---|
| Q2 | 45,988 | 0.940 | 0.983 | 0.062 | 0.205 |
| Q4 (>$225,000) | 45,480 | 0.790 | 0.964 | 0.071 | 0.095 |
| Q3 | 45,125 | 0.871 | 0.971 | 0.072 | 0.133 |
| Q1 (<=$35,000) | 42,841 | 0.934 | 0.980 | 0.070 | 0.274 |
