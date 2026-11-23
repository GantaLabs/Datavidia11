"""
src/metrics.py - Competition Evaluation Metrics (with Pluggable Custom Metric Hook)
"""

import importlib.util
from pathlib import Path
import numpy as np
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    log_loss,
    accuracy_score,
    f1_score,
    mean_squared_error,
    mean_absolute_error,
    r2_score
)


def _load_custom_metric():
    custom_path = Path("src/custom_metric.py")
    if not custom_path.exists():
        return None
    spec = importlib.util.spec_from_file_location("custom_metric", str(custom_path))
    if spec and spec.loader:
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return getattr(mod, "compute_custom_metric", None)
    return None


def _probs_to_labels(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    if y_pred.ndim > 1:
        return np.argmax(y_pred, axis=1)
    unique_classes = np.unique(y_true)
    if len(unique_classes) == 2 and np.issubdtype(y_pred.dtype, np.floating) and (y_pred.max() <= 1.0) and (y_pred.min() >= 0.0):
        return (y_pred >= 0.5).astype(int)
    return np.round(y_pred).astype(int)


def compute_metric(y_true, y_pred, metric_name: str = "roc_auc") -> float:
    """
    Computes competition evaluation metrics.
    Supports: roc_auc, pr_auc, log_loss, accuracy, f1_macro, f1_weighted, f1_micro,
    f1_binary, rmse, mae, r2, rmsle, and custom (via src/custom_metric.py).
    """
    name = metric_name.lower().replace("-", "_").replace(" ", "_")
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    if name == "custom":
        fn = _load_custom_metric()
        if fn is None:
            raise FileNotFoundError("metric_name='custom' requires src/custom_metric.py with compute_custom_metric(y_true, y_pred)")
        return float(fn(y_true, y_pred))

    if name in ["roc_auc", "auc"]:
        if len(np.unique(y_true)) == 2:
            if y_pred.ndim > 1 and y_pred.shape[1] == 2:
                y_pred = y_pred[:, 1]
            return float(roc_auc_score(y_true, y_pred))
        return float(roc_auc_score(y_true, y_pred, multi_class="ovr"))

    if name in ["pr_auc", "average_precision"]:
        if y_pred.ndim > 1 and y_pred.shape[1] == 2:
            y_pred = y_pred[:, 1]
        return float(average_precision_score(y_true, y_pred))

    if name in ["log_loss", "logloss"]:
        return float(log_loss(y_true, y_pred))

    if name in ["accuracy", "acc"]:
        return float(accuracy_score(y_true, _probs_to_labels(y_true, y_pred)))

    if name in ["f1_macro", "macro_f1"]:
        return float(f1_score(y_true, _probs_to_labels(y_true, y_pred), average="macro", zero_division=0))

    if name in ["f1_weighted", "weighted_f1"]:
        return float(f1_score(y_true, _probs_to_labels(y_true, y_pred), average="weighted", zero_division=0))

    if name in ["f1_micro", "micro_f1"]:
        return float(f1_score(y_true, _probs_to_labels(y_true, y_pred), average="micro", zero_division=0))

    if name in ["f1", "f1_binary"]:
        avg = "binary" if len(np.unique(y_true)) <= 2 else "macro"
        return float(f1_score(y_true, _probs_to_labels(y_true, y_pred), average=avg, zero_division=0))

    if name == "rmse":
        return float(np.sqrt(mean_squared_error(y_true, y_pred)))

    if name == "rmsle":
        y_true_clip = np.clip(y_true, 0, None)
        y_pred_clip = np.clip(y_pred, 0, None)
        return float(np.sqrt(mean_squared_error(np.log1p(y_true_clip), np.log1p(y_pred_clip))))

    if name == "mae":
        return float(mean_absolute_error(y_true, y_pred))

    if name == "r2":
        return float(r2_score(y_true, y_pred))

    fn = _load_custom_metric()
    if fn is not None:
        return float(fn(y_true, y_pred))

    try:
        return float(accuracy_score(y_true, _probs_to_labels(y_true, y_pred)))
    except Exception:
        return float(np.sqrt(mean_squared_error(y_true, y_pred)))
