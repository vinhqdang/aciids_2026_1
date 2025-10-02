"""
Evaluation metrics for fraud detection.
"""

import torch
import numpy as np
from sklearn.metrics import (
    roc_auc_score, average_precision_score, precision_recall_curve,
    precision_score, recall_score, f1_score, confusion_matrix
)
from typing import Dict, Tuple


def compute_metrics(y_true: np.ndarray,
                   y_scores: np.ndarray,
                   threshold: float = 0.5,
                   k_list: list = [100, 500, 1000]) -> Dict[str, float]:
    """
    Compute comprehensive fraud detection metrics.

    Args:
        y_true: True binary labels
        y_scores: Predicted probabilities
        threshold: Classification threshold
        k_list: List of k values for precision@k

    Returns:
        Dictionary of metrics
    """
    metrics = {}

    # ROC-AUC
    try:
        metrics['roc_auc'] = roc_auc_score(y_true, y_scores)
    except:
        metrics['roc_auc'] = 0.0

    # Average Precision (AUPRC)
    try:
        metrics['auprc'] = average_precision_score(y_true, y_scores)
    except:
        metrics['auprc'] = 0.0

    # Binary predictions
    y_pred = (y_scores >= threshold).astype(int)

    # Precision, Recall, F1
    metrics['precision'] = precision_score(y_true, y_pred, zero_division=0)
    metrics['recall'] = recall_score(y_true, y_pred, zero_division=0)
    metrics['f1'] = f1_score(y_true, y_pred, zero_division=0)

    # Confusion matrix
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    metrics['true_positives'] = tp
    metrics['false_positives'] = fp
    metrics['true_negatives'] = tn
    metrics['false_negatives'] = fn

    # False Positive Rate
    metrics['fpr'] = fp / (fp + tn + 1e-8)

    # Precision@k
    for k in k_list:
        if len(y_scores) >= k:
            top_k_idx = np.argsort(y_scores)[-k:]
            precision_at_k = y_true[top_k_idx].sum() / k
            metrics[f'precision@{k}'] = precision_at_k

    return metrics


def compute_calibration_error(y_true: np.ndarray,
                              y_scores: np.ndarray,
                              n_bins: int = 10) -> float:
    """
    Compute Expected Calibration Error (ECE).
    """
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]

    ece = 0.0
    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        in_bin = (y_scores > bin_lower) & (y_scores <= bin_upper)
        prop_in_bin = in_bin.mean()

        if prop_in_bin > 0:
            accuracy_in_bin = y_true[in_bin].mean()
            avg_confidence_in_bin = y_scores[in_bin].mean()
            ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin

    return ece
