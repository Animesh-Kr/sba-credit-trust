"""Evaluation metrics for imbalanced, high-stakes binary classification.

Accuracy is deliberately *not* the headline. For ~18%-prevalence default prediction the honest
metrics are PR-AUC (ranking under imbalance), ROC-AUC, minority-class F1/recall, and — central to
this project's trustworthiness thesis — probabilistic calibration via Brier score and Expected
Calibration Error (ECE).
"""
from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def expected_calibration_error(
    y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 15
) -> float:
    """Equal-width-binned ECE: sum_b (|bin| / N) * |acc(b) - conf(b)|.

    ``y_prob`` is P(default). Returns a value in [0, 1]; lower is better-calibrated.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_prob = np.asarray(y_prob, dtype=float)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    n = len(y_true)
    for b, (lo, hi) in enumerate(zip(bins[:-1], bins[1:])):
        # first bin closed on the left (include prob == 0.0); last bin closed on the right
        # (include prob == 1.0); interior bins are (lo, hi].
        lower = (y_prob >= lo) if b == 0 else (y_prob > lo)
        in_bin = lower & (y_prob <= hi)
        count = int(in_bin.sum())
        if count == 0:
            continue
        conf = float(y_prob[in_bin].mean())
        acc = float(y_true[in_bin].mean())
        ece += (count / n) * abs(acc - conf)
    return ece


def classification_report(
    y_true: np.ndarray, y_prob: np.ndarray, threshold: float = 0.5, n_bins: int = 15
) -> dict[str, float]:
    """Threshold-free ranking + probabilistic metrics plus point metrics at ``threshold``.

    The positive class is *default* (1). ``f1_minority`` is the F1 of the default class, which
    is the rare, costly one we most care about.
    """
    y_true = np.asarray(y_true, dtype=int)
    y_prob = np.asarray(y_prob, dtype=float)
    y_pred = (y_prob >= threshold).astype(int)
    return {
        "pr_auc": float(average_precision_score(y_true, y_prob)),
        "roc_auc": float(roc_auc_score(y_true, y_prob)),
        "f1_minority": float(f1_score(y_true, y_pred, pos_label=1, zero_division=0)),
        "precision_minority": float(precision_score(y_true, y_pred, pos_label=1, zero_division=0)),
        "recall_minority": float(recall_score(y_true, y_pred, pos_label=1, zero_division=0)),
        "brier": float(brier_score_loss(y_true, y_prob)),
        "ece": float(expected_calibration_error(y_true, y_prob, n_bins=n_bins)),
        "prevalence": float(y_true.mean()),
        "threshold": float(threshold),
    }
