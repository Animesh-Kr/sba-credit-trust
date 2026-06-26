"""Serving bundle: everything the app/ONNX needs to turn a loan into a trustworthy decision.

A bundle = the trained pipeline + an isotonic calibrator + the decision thresholds + the form
schema + metadata.

**Decision policy (deployment).** The app makes *per-loan* decisions on the isotonic-calibrated
probability, which is intuitive and inspectable:
  * auto-approve if P(default) <= alpha            (each approved loan's own risk is below target)
  * auto-reject  if P(default) >= reject_threshold (more likely to default than not)
  * else refer to a human underwriter

This is intentionally stricter than the conformal *marginal* risk guarantee studied in the research
reports (which controls the average default rate among approved, and can therefore admit a few
moderate-risk loans). For a tool a human inspects, per-loan thresholds are the honest choice.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field

import numpy as np
import pandas as pd


@dataclass
class ServingBundle:
    pipeline: object                      # sklearn Pipeline (prep + XGBClassifier)
    calibrator: object                    # IsotonicCalibrator fitted on a held-out slice
    reject_threshold: float               # auto-reject if calibrated P(default) >= this (e.g. 0.5)
    alphas: list                          # selectable approve targets, e.g. [0.05, 0.10]
    numeric_cols: list                    # feature schema for the UI
    categorical_cols: list
    choices: dict                         # categorical col -> allowed values
    numeric_defaults: dict                # col -> median (UI default)
    metadata: dict = field(default_factory=dict)

    # --- inference ---
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Isotonic-calibrated P(default)."""
        raw = self.pipeline.predict_proba(X)[:, 1]
        return np.asarray(self.calibrator.transform(raw), dtype=float)

    def decide(self, prob, alpha: float = 0.05) -> np.ndarray:
        """Per-loan decision array of {'auto-approve','auto-reject','refer to human'}."""
        prob = np.atleast_1d(np.asarray(prob, dtype=float))
        out = np.full(len(prob), "refer to human", dtype=object)
        out[prob <= alpha] = "auto-approve"
        out[prob >= self.reject_threshold] = "auto-reject"
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
