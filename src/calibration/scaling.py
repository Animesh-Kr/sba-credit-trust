"""Post-hoc probability calibration: temperature scaling and isotonic regression.

Both are fit on a held-out **calibration** split (disjoint from train and test) and applied to
test. Temperature scaling (Guo et al., 2017) learns a single scalar T that divides the logits;
it preserves the model's ranking (AUC unchanged) while improving probability calibration.
Isotonic regression is a non-parametric monotone alternative.
"""
from __future__ import annotations

import numpy as np


class TemperatureScaler:
    """Learn scalar T>0 minimising NLL on calibration logits, then p = sigmoid(logit / T)."""

    def __init__(self):
        self.T_: float = 1.0

    def fit(self, logits: np.ndarray, labels: np.ndarray) -> TemperatureScaler:
        import torch

        z = torch.from_numpy(np.asarray(logits, dtype=np.float64))
        y = torch.from_numpy(np.asarray(labels, dtype=np.float64))
        log_T = torch.zeros(1, dtype=torch.float64, requires_grad=True)
        opt = torch.optim.LBFGS([log_T], lr=0.1, max_iter=100)

        def closure():
            opt.zero_grad()
            T = torch.exp(log_T)  # keep T>0
            loss = torch.nn.functional.binary_cross_entropy_with_logits(z / T, y)
            loss.backward()
            return loss

        opt.step(closure)
        self.T_ = float(torch.exp(log_T).item())
        return self

    def transform(self, logits: np.ndarray) -> np.ndarray:
        from scipy.special import expit

        return expit(np.asarray(logits, dtype=np.float64) / self.T_)


class IsotonicCalibrator:
    """Monotone mapping from raw probability to calibrated probability."""

    def __init__(self):
        from sklearn.isotonic import IsotonicRegression

        self.iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)

    def fit(self, prob: np.ndarray, labels: np.ndarray) -> IsotonicCalibrator:
        self.iso.fit(np.asarray(prob, dtype=np.float64), np.asarray(labels, dtype=np.float64))
        return self

    def transform(self, prob: np.ndarray) -> np.ndarray:
        return self.iso.predict(np.asarray(prob, dtype=np.float64))


def logit(p: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """Inverse-sigmoid, for calibrating models that expose probabilities not logits (e.g. XGB)."""
    p = np.clip(np.asarray(p, dtype=np.float64), eps, 1 - eps)
    return np.log(p / (1 - p))
