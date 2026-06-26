# SBA Credit-Trust — E-1 Rigorous + E-9 Adaptive Conformal

Target α = 0.05. Train ≤ 2006; future > 2006. Runtime 41.4s.

## E-1 — Rigorous (union-bounded) vs heuristic static threshold

| Method | t_lo | in-era default@approved | future default@approved | future coverage |
|---|---|---|---|---|
| heuristic | 0.927 | 0.0494 | 0.1067 | 0.817 |
| rigorous | 0.920 | 0.0476 | 0.1017 | 0.719 |

> The rigorous (Bonferroni union-bounded) threshold is more conservative in-era; both still drift above α on the shifted future — motivating E-9.

## E-9 — Static vs Adaptive Conformal Inference, online by approval date (γ=0.1)

Future loans streamed in approval-date order (150 batches); per-year aggregation below.

| Year | static default@approved | static coverage | **ACI default@approved** | ACI coverage | ACI threshold |
|---|---|---|---|---|---|
| 2007 | 0.125 | 0.627 | **0.054** | 0.555 | 0.792 |
| 2008 | 0.150 | 0.662 | **0.049** | 0.539 | 0.702 |
| 2009 | 0.121 | 0.870 | **0.062** | 0.731 | 0.644 |
| 2010 | 0.095 | 0.922 | **0.042** | 0.753 | 0.527 |
| 2011 | 0.037 | 0.923 | **0.025** | 0.867 | 0.812 |
| 2012 | 0.020 | 0.924 | **0.024** | 0.933 | 0.944 |
| 2013 | 0.018 | 0.939 | **0.020** | 0.968 | 0.963 |

> **Long-run default rate among approved:** static = **0.1106**, adaptive = **0.0474** (target 0.05). ACI tracks the target by tightening the threshold after high-default batches — turning Phase D's negative result (static conformal breaks under shift) into a positive one.
