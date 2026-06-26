# SBA Credit-Trust — Phase E Deployment

Served model: **XGBoost (Optuna-tuned) + temperature calibration + conformal selection**.

## Model artefact

- Test PR-AUC **0.9263**, ROC-AUC 0.9815, ECE 0.0559, Brier 0.0456.
- Temperature T = 0.955. Conformal thresholds: {0.05: {'t_lo': 0.918, 't_hi': 0.974}, 0.1: {'t_lo': 0.983, 't_hi': 0.851}}.

## ONNX export + latency

- ONNX model size: **5713 KB**.
- Parity (raw prob, n=5,000): max|Δ| = **5.96e-07** (PASS).
- Single-row latency: sklearn **2.258 ms**, ONNX **0.040 ms**.

> The deployed app applies temperature calibration + the conformal decision on top of the model probability, giving a calibrated risk + auto-approve/reject/refer outcome.
