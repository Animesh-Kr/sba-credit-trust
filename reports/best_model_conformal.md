# SBA Credit-Trust — E-8 Best Model (FT-Transformer) + Calibration + Conformal

Test n = 179,434. Runtime 689.1s. The strongest DL model now carries the calibration + conformal stack (previously on the MLP).

## Calibration
| Variant | PR-AUC | ROC-AUC | F1(def) | Brier | ECE |
|---|---|---|---|---|---|
| FT-Transformer raw | 0.8985 | 0.9722 | 0.7984 | 0.0769 | 0.1592 |
| FT-Transformer + isotonic | 0.8952 | 0.9722 | 0.8239 | 0.0447 | 0.0020 |

## Conformal selective prediction (isotonic-calibrated FT-Transformer)

| α | method | coverage | abstain | **default@approved** |
|---|---|---|---|---|
| 0.05 | heuristic | 0.940 | 0.060 | **0.0497** |
| 0.05 | rigorous | 0.845 | 0.155 | **0.0473** |
| 0.10 | heuristic | 0.943 | 0.057 | **0.0565** |
| 0.10 | rigorous | 0.909 | 0.091 | **0.0974** |

> The best DL model, calibrated, holds the selective-risk target on the test split. (Formal caveat: AUDIT M-1; temporal robustness: see adaptive_conformal report.)
