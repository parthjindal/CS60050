import numpy as np
from typing import List, Dict, Tuple


def get_metrics(
    y_pred=None,
    y_true=None,
    metrics: List[str] = ["Accuracy"],
    classes: List[str] = ["Ham", "Spam"]
) -> Dict:

    if isinstance(y_pred, np.ndarray) == False:
        y_pred = y_pred.to_numpy()
    if isinstance(y_true, np.ndarray) == False:
        y_true = y_true.to_numpy()

    results = {}

    for metric in metrics:
        if metric == "Accuracy":
            results[metric] = accuracy(y_pred, y_true)
        elif metric == "Precision":
            results[metric] = precision(y_pred, y_true)
        elif metric == "Recall":
            results[metric] = recall(y_pred, y_true)
        elif metric == "F1":
            results[metric] = f1_score(y_pred, y_true)
        elif metric == "Confusion Matrix":
            results[metric] = confusion_matrix(y_pred, y_true, classes)
        else:
            raise ValueError("Invalid Metric")

    return results


def accuracy(y_pred: np.ndarray, y_true: np.ndarray) -> float:
    return np.mean(y_pred == y_true)


def precision(y_pred: np.ndarray, y_true: np.ndarray) -> float:
    """
    Computes the precision score
    Precision score = true_pos / (true_pos + false_pos)
    Args:
        y_true: The true labels.
        y_pred: The predicted labels.
    Returns:
        Precision Score.
    """
    true_pos = np.sum(y_pred * y_true)
    return true_pos / (np.sum(y_pred) + 1e-8)


def recall(y_pred: np.ndarray, y_true: np.ndarray) -> float:
    """
    Computes the recall score
    Recall score = true_pos / (true_pos + false_neg)
    Args:
        y_true: The true labels.
        y_pred: The predicted labels.
    Returns:
        Recall Score.
    """
    true_pos = np.sum(y_pred * y_true)
    return true_pos / (np.sum(y_true))


def f1_score(y_pred: np.ndarray, y_true: np.ndarray) -> float:
    """
    Computes the recall score for a given set of labels.
    Recall score = true_pos / (true_pos + false_neg)
    Args:
        y_true: The true labels.
        y_pred: The predicted labels.
        label: The label to consider.
    Returns:
        Recall Score.
    """
    p = precision(y_pred, y_true)
    r = recall(y_pred, y_true)
    return 2 * (p * r) / (p + r)


def confusion_matrix(y_pred: np.ndarray, y_true: np.ndarray, classes: List[str]) -> np.ndarray:
    confusion_matrix = np.zeros((len(classes), len(classes)))
    for i in range(len(y_pred)):
        confusion_matrix[y_true[i]][y_pred[i]] += 1
    return confusion_matrix
