from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score


@dataclass
class ClassificationMetrics:
    accuracy: float
    precision: float
    recall: float
    f1: float
    confusion: np.ndarray
    precision_per_class: list[float] | None = None
    recall_per_class: list[float] | None = None


def compute_metrics(
    y_true: Sequence[int],
    y_pred: Sequence[int],
    average: str = 'weighted',
    labels: Sequence[int] | None = None,
) -> ClassificationMetrics:
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    if labels is None:
        labels = sorted(np.unique(np.concatenate([y_true, y_pred])))
    labels = np.asarray(labels)
    n_classes = len(labels)
    confusion = confusion_matrix(y_true, y_pred, labels=labels)

    prec = precision_score(y_true, y_pred, average=average, zero_division=0, labels=labels)
    rec = recall_score(y_true, y_pred, average=average, zero_division=0, labels=labels)
    f1 = f1_score(y_true, y_pred, average=average, zero_division=0, labels=labels)

    if n_classes <= 10:
        prec_per = precision_score(y_true, y_pred, average=None, zero_division=0, labels=labels).tolist()
        rec_per = recall_score(y_true, y_pred, average=None, zero_division=0, labels=labels).tolist()
    else:
        prec_per = rec_per = None

    return ClassificationMetrics(
        accuracy=float(accuracy_score(y_true, y_pred)),
        precision=float(prec),
        recall=float(rec),
        f1=float(f1),
        confusion=confusion,
        precision_per_class=prec_per,
        recall_per_class=rec_per,
    )
