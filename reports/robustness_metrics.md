# SBA Credit-Trust — E-2 Metric Uncertainty (XGBoost)

Addresses AUDIT M-2. Runtime **222.5s**.

## Across-seed variability (5 seeds: [42, 7, 13, 21, 100])

| Metric | mean | std | min | max |
|---|---|---|---|---|
| pr_auc | 0.9132 | 0.0022 | 0.9093 | 0.9150 |
| roc_auc | 0.9785 | 0.0003 | 0.9781 | 0.9789 |
| f1_minority | 0.8185 | 0.0018 | 0.8157 | 0.8200 |
| brier | 0.0536 | 0.0007 | 0.0532 | 0.0548 |
| ece | 0.0745 | 0.0010 | 0.0737 | 0.0763 |

## Bootstrap 95% CIs (reference seed, test n=179,434, 1000 resamples)

| Metric | point | 95% CI |
|---|---|---|
| pr_auc | 0.9139 | [0.9107, 0.9169] |
| roc_auc | 0.9785 | [0.9777, 0.9792] |
| brier | 0.0533 | [0.0526, 0.0540] |
| ece | 0.0741 | [0.0730, 0.0751] |

> Across-seed std captures train/split variance; the bootstrap CI captures finite-test-set variance. Both are small, so the headline XGBoost numbers are stable.
