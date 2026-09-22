"""Test-set metrics, calibration, confidence intervals and significance tests."""

import numpy as np
from scipy.stats import binomtest
from sklearn.metrics import (
    accuracy_score, confusion_matrix, log_loss, precision_recall_fscore_support,
    top_k_accuracy_score,
)


def expected_calibration_error(y_true, probs, n_bins=15):
    """ECE: weighted mean |accuracy - confidence| over equal-width confidence bins."""
    conf = probs.max(axis=1)
    correct = probs.argmax(axis=1) == y_true
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        in_bin = (conf > lo) & (conf <= hi)
        if in_bin.any():
            ece += in_bin.mean() * abs(correct[in_bin].mean() - conf[in_bin].mean())
    return float(ece)


def classification_metrics(y_true, probs):
    """Scalar metrics computed from predicted class probabilities."""
    labels = np.arange(probs.shape[1])
    y_pred = probs.argmax(axis=1)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, average="macro", zero_division=0)
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "top3_accuracy": float(top_k_accuracy_score(y_true, probs, k=3, labels=labels)),
        "macro_precision": float(precision),
        "macro_recall": float(recall),
        "macro_f1": float(f1),
        "nll": float(log_loss(y_true, np.clip(probs, 1e-7, 1.0), labels=labels)),
        "ece": expected_calibration_error(y_true, probs),
    }


def per_class_f1(y_true, probs):
    labels = np.arange(probs.shape[1])
    _, _, f1, _ = precision_recall_fscore_support(
        y_true, probs.argmax(axis=1), labels=labels, average=None, zero_division=0)
    return f1.tolist()


def confusion(y_true, probs):
    return confusion_matrix(y_true, probs.argmax(axis=1),
                            labels=np.arange(probs.shape[1])).tolist()


def bootstrap_accuracy_ci(correct, n_boot=2000, alpha=0.05, seed=0):
    """Percentile bootstrap CI for accuracy over the test set."""
    rng = np.random.default_rng(seed)
    correct = np.asarray(correct, dtype=float)
    idx = rng.integers(0, len(correct), size=(n_boot, len(correct)))
    boots = correct[idx].mean(axis=1)
    return float(np.quantile(boots, alpha / 2)), float(np.quantile(boots, 1 - alpha / 2))


def mcnemar_p(correct_a, correct_b):
    """Exact McNemar test on paired per-sample correctness of two models."""
    correct_a, correct_b = np.asarray(correct_a, bool), np.asarray(correct_b, bool)
    b = int((correct_a & ~correct_b).sum())
    c = int((~correct_a & correct_b).sum())
    if b + c == 0:
        return 1.0
    return float(binomtest(b, b + c, 0.5).pvalue)
