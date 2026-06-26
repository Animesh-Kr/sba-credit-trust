"""Export the serving pipeline to ONNX, check parity, and benchmark latency.

ONNX gives a portable, fast inference artefact (matches the deployment pattern of the author's other
projects). Temperature calibration is a scalar post-processing step applied identically in Python and
after ONNX, so the ONNX graph carries the raw model and parity is checked on raw probabilities.
"""
from __future__ import annotations

import os
import time

import numpy as np

from ..data import clean, split
from .bundle import DEFAULT_PATH, ServingBundle

REPORT = os.path.join(os.path.dirname(__file__), "..", "..", "reports", "deployment_metrics.md")
ONNX_PATH = os.path.join(os.path.dirname(__file__), "..", "..", "models", "sba_xgb.onnx")


def _register_xgboost():
    """Teach skl2onnx how to convert XGBClassifier (via onnxmltools)."""
    from onnxmltools.convert.xgboost.operator_converters.XGBoost import convert_xgboost
    from skl2onnx import update_registered_converter
    from skl2onnx.common.shape_calculator import calculate_linear_classifier_output_shapes
    from xgboost import XGBClassifier

    update_registered_converter(
        XGBClassifier, "XGBoostXGBClassifier",
        calculate_linear_classifier_output_shapes, convert_xgboost,
        options={"nocl": [True, False], "zipmap": [True, False, "columns"]},
    )


def run(raw_csv, bundle_path=DEFAULT_PATH, n_bench=5000, seed=42, nrows=None):
    import onnxruntime as ort
    from skl2onnx import to_onnx
    from skl2onnx.common.data_types import FloatTensorType

    _register_xgboost()
    b = ServingBundle.load(bundle_path)
    X, y = clean.load_clean(raw_csv, nrows=nrows)
    _, te = split.stratified_split(y, test_size=0.2, seed=seed)
    Xs = X.iloc[te[:n_bench]]

    # Preprocessing (string imputation + one-hot) is not ONNX-friendly, so we keep it in the
    # serving layer and export the gradient-boosted MODEL over the transformed numeric features.
    prep = b.pipeline.named_steps["prep"]
    clf = b.pipeline.named_steps["clf"]

    def to_dense(df):
        Xt = prep.transform(df)
        return np.asarray(Xt.toarray() if hasattr(Xt, "toarray") else Xt, dtype=np.float32)

    Xt_s = to_dense(Xs)
    n_features = Xt_s.shape[1]
    onnx_model = to_onnx(clf, initial_types=[("input", FloatTensorType([None, n_features]))],
                         options={id(clf): {"zipmap": False}},
                         target_opset={"": 17, "ai.onnx.ml": 3})
    os.makedirs(os.path.dirname(os.path.abspath(ONNX_PATH)), exist_ok=True)
    with open(ONNX_PATH, "wb") as f:
        f.write(onnx_model.SerializeToString())
    size_kb = os.path.getsize(ONNX_PATH) / 1024

    sess = ort.InferenceSession(ONNX_PATH, providers=["CPUExecutionProvider"])
    in_name = sess.get_inputs()[0].name

    def onnx_proba(Xt):
        out = sess.run(None, {in_name: np.asarray(Xt, dtype=np.float32)})
        probs = out[-1]
        return np.asarray(probs)[:, 1] if np.ndim(probs) == 2 else np.asarray(probs).ravel()

    # parity on raw model probabilities (over the transformed features)
    sk_raw = clf.predict_proba(Xt_s)[:, 1]
    try:
        ox_raw = onnx_proba(Xt_s)
        max_diff = float(np.max(np.abs(sk_raw - ox_raw)))
        parity_ok = max_diff < 1e-3
    except Exception as e:  # noqa: BLE001
        max_diff, parity_ok = float("nan"), False
        print(f"[onnx] parity check failed: {e}")

    # latency: single-row model inference, sklearn vs onnx (preprocessing excluded, same for both)
    one = Xt_s[:1]
    sk_t = _bench(lambda: clf.predict_proba(one))
    ox_t = _bench(lambda: onnx_proba(one)) if parity_ok else float("nan")

    out = _write(b, size_kb, max_diff, parity_ok, sk_t, ox_t, len(Xs))
    print(f"[onnx] wrote {out}  (ONNX {size_kb:.0f} KB, parity max-abs-diff={max_diff:.2e})")
    return out


def _bench(fn, n=200):
    fn()  # warmup
    t0 = time.perf_counter()
    for _ in range(n):
        fn()
    return (time.perf_counter() - t0) / n * 1000  # ms/call


def _write(b, size_kb, max_diff, parity_ok, sk_ms, ox_ms, n):
    m = b.metadata
    L = ["# SBA Credit-Trust — Phase E Deployment\n"]
    L.append(f"Served model: **{m.get('model','')}**.\n")
    L.append("## Model artefact\n")
    L.append(f"- Test PR-AUC **{m.get('test_pr_auc'):.4f}**, ROC-AUC {m.get('test_roc_auc'):.4f}, "
             f"ECE {m.get('test_ece'):.4f}, Brier {m.get('test_brier'):.4f}.")
    L.append(f"- Temperature T = {m.get('temperature'):.3f}. Conformal thresholds: "
             f"{ {a: {k: round(v,3) for k,v in t.items()} for a,t in b.conformal.items()} }.\n")
    L.append("## ONNX export + latency\n")
    L.append(f"- ONNX model size: **{size_kb:.0f} KB**.")
    L.append(f"- Parity (raw prob, n={n:,}): max|Δ| = **{max_diff:.2e}** "
             f"({'PASS' if parity_ok else 'see note'}).")
    L.append(f"- Single-row latency: sklearn **{sk_ms:.3f} ms**" +
             (f", ONNX **{ox_ms:.3f} ms**." if parity_ok else "."))
    L.append("\n> The deployed app applies temperature calibration + the conformal decision on top of "
             "the model probability, giving a calibrated risk + auto-approve/reject/refer outcome.\n")
    os.makedirs(os.path.dirname(os.path.abspath(REPORT)), exist_ok=True)
    out = os.path.abspath(REPORT)
    with open(out, "w", encoding="utf-8") as f:
        f.write("\n".join(L))
    return out


if __name__ == "__main__":
    import sys

    csv = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "data", "SBAnational.csv"))
    nrows = int(sys.argv[1]) if len(sys.argv) > 1 else None
    run(csv, nrows=nrows)
