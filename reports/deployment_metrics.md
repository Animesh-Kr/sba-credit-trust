# SBA Credit-Trust — Phase E Deployment

Served model: **XGBoost (Optuna-tuned) + isotonic calibration + per-loan decision**.

## Model artefact

- Test PR-AUC **0.9235**, ROC-AUC 0.9814, ECE 0.0012, Brier 0.0350.
- Isotonic-calibrated; per-loan policy (reject ≥ 50%). Decision validation: {0.05: {'approve_rate': 0.7145412798020442, 'default_among_approved': 0.006465802999695819, 'reject_rate': 0.17009596843407604, 'refer_rate': 0.11536275176387975}, 0.1: {'approve_rate': 0.7621799658927516, 'default_among_approved': 0.010668246064301957, 'reject_rate': 0.17009596843407604, 'refer_rate': 0.06772406567317231}}.

## ONNX export + latency

- ONNX model size: **5713 KB**.
- Parity (raw prob, n=5,000): max|Δ| = **5.96e-07** (PASS).
- Single-row latency: sklearn **2.073 ms**, ONNX **0.027 ms**.

> The deployed app applies temperature calibration + the conformal decision on top of the model probability, giving a calibrated risk + auto-approve/reject/refer outcome.
