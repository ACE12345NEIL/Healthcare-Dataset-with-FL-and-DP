"""Metrics shared by every model so all are scored identically on the held-out test set."""
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             precision_score, recall_score, roc_auc_score)


METRIC_NAMES = ["accuracy", "precision", "recall", "f1", "roc_auc"]


def compute_metrics(y_true, y_pred, y_score) -> dict:
    """Confusion matrix layout: [[TN, FP], [FN, TP]] (labels 0 then 1)."""
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, y_score)),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=[0, 1]).tolist(),
    }


def save_json(obj, path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    def default(o):
        if isinstance(o, np.generic):
            return o.item()
        if isinstance(o, np.ndarray):
            return o.tolist()
        raise TypeError(f"Not JSON serializable: {type(o)}")

    path.write_text(json.dumps(obj, indent=2, default=default))
    return path


def format_metrics_row(name: str, m: dict) -> str:
    return (f"{name:<22}{m['accuracy']:>9.3f}{m['precision']:>10.3f}"
            f"{m['recall']:>8.3f}{m['f1']:>8.3f}{m['roc_auc']:>9.3f}")


METRICS_HEADER = f"{'Model':<22}{'Accuracy':>9}{'Precision':>10}{'Recall':>8}{'F1':>8}{'ROC-AUC':>9}"
