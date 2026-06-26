"""Serving bundle: everything the app/ONNX needs to turn a loan into a trustworthy decision.

A bundle = the trained pipeline + a temperature calibrator + conformal approve/reject thresholds +
the form schema (so the UI can render inputs) + metadata. Calibration uses temperature scaling (a
single scalar) so it is trivial to apply both in Python and after ONNX export.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy.special import expit
from scipy.special import logit as _logit


@dataclass
class ServingBundle:
    pipeline: object                      # sklearn Pipeline (prep + XGBClassifier)
    temperature: float                    # calibration scalar: p = sigmoid(logit(p_raw)/T)
    conformal: dict                       # alpha -> {"t_lo": float, "t_hi": float}
    numeric_cols: list                    # feature schema for the UI
    categorical_cols: list
    choices: dict                         # categorical col -> list of allowed values
    numeric_defaults: dict                # col -> median (UI default)
    metadata: dict = field(default_factory=dict)

    # --- inference ---
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        raw = self.pipeline.predict_proba(X)[:, 1]
        return expit(_logit(np.clip(raw, 1e-6, 1 - 1e-6)) / self.temperature)

    def decide(self, prob: np.ndarray, alpha: float = 0.05) -> np.ndarray:
        t = self.conformal[alpha]
        prob = np.asarray(prob, dtype=float)
        out = np.full(len(prob), "abstain / refer", dtype=object)
        out[(prob <= t["t_lo"]) & (prob < t["t_hi"])] = "auto-approve"
        out[(prob >= t["t_hi"]) & (prob > t["t_lo"])] = "auto-reject"
        return out

    # --- persistence ---
    def save(self, path: str) -> str:
        import joblib

        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        joblib.dump(self, path)
        return os.path.abspath(path)

    @staticmethod
    def load(path: str) -> ServingBundle:
        import joblib

        return joblib.load(path)


DEFAULT_PATH = os.path.join(os.path.dirname(__file__), "..", "..", "models", "serving_bundle.joblib")
