# SBA Credit-Trust — E-4 Real OOD Benchmark (leave-one-sector-out)

Held-out NAICS sector **52** removed from training (n_OOD=9,470); Mahalanobis detector fit on the remaining sectors. In-distribution test n=177,540. Runtime 239.7s.

| OOD type | AUROC |
|---|---|
| **Real (unseen sector 52)** | **0.4728** |
| Synthetic (feature permutation, Phase-C style) | 0.8084 |

- Median Mahalanobis distance: in-distribution = 102.6, unseen-sector = 98.3.

> A held-out industry is a genuine covariate shift; AUROC > 0.5 means the penultimate-layer Mahalanobis score flags it without ever being trained to. This replaces the synthetic-only OOD evidence flagged in AUDIT M-4.
