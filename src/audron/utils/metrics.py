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


def compute_metrics(y_true: Sequence[int], y_pred: Sequence[int], average: str = 'weighted') -> ClassificationMetrics:
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    return ClassificationMetrics(
        accuracy=float(accuracy_score(y_true, y_pred)),
        precision=float(precision_score(y_true, y_pred, average=average, zero_division=0)),
        recall=float(recall_score(y_true, y_pred, average=average, zero_division=0)),
        f1=float(f1_score(y_true, y_pred, average=average, zero_division=0)),
        confusion=confusion_matrix(y_true, y_pred),
    )
