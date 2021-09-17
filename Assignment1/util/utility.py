from typing import List, Tuple
import numpy as np
import pandas as pd


def gini_index(targets) -> float:
    """
    Computes the gini index for a given target labels.
    Arguments:
        targets: A numpy array of shape (n_samples, )
    Returns:
        The gini index for the given target labels.
    """
    label_count = np.bincount(targets)
    return 1 - np.sum((label_count / len(targets)) ** 2)


def entropy(targets) -> float:
    """
    Computes the entropy for a given target labels.
    Arguments:
        targets: A numpy array of shape (n_samples, )
    Returns:
        The entropy for the given target labels.
    """
    label_count = np.bincount(targets).astype(float)
    label_count += 1e-7
    return -np.sum((label_count / len(targets)) * np.log2(label_count / len(targets)))


def gini_gain(data: pd.DataFrame, attribute: str, targets: pd.Series) -> float:
    """
    Computes the gini gain for a given attribute and target labels.
    Arguments:
        data: A numpy array of shape (n_samples, n_features)
        attribute: Attribute name to compute the gini gain for
        targets: A numpy array of shape (n_samples, )
    Returns:
        The gini gain for the given attribute and target labels.
    """
    init_gini = gini_index(targets)  # Initial gini index
    # All possible values of the attribute
    attribute_values = np.unique(data.iloc[:][attribute])
    new_gini = 0  # Initialize the new gini index

    for value in attribute_values:  # For each value of the attribute
        # Get the data for the current value
        sub_data = data.iloc[:][data.iloc[:][attribute] == value]
        # Get the targets for the current value
        sub_targets = targets.iloc[:][data.iloc[:][attribute] == value]
        sub_gini = gini_index(sub_targets)
        new_gini += len(sub_data) / len(data) * sub_gini
    return init_gini - new_gini  # Return the gini gain


def information_gain(data: pd.DataFrame, attribute: str, targets: pd.Series) -> float:
    """
    Computes the information gain for a given attribute and target labels.
    Arguments:
        data: A numpy array of shape (n_samples, n_features)
        attribute: Attribute name to compute the information gain for
        targets: A numpy array of shape (n_samples, )
    Returns:
        The information gain for the given attribute and target labels.
    """
    init_entropy = entropy(targets)
    attribute_values = np.unique(data.iloc[:][attribute])
    new_entropy = 0
    for value in attribute_values:
        # Get the data for the current value
        sub_data = data.iloc[:][data.iloc[:][attribute] == value]
        # Get the targets for the current value
        sub_targets = targets.iloc[:][data.iloc[:][attribute] == value]
        sub_entropy = entropy(sub_targets)
        new_entropy += len(sub_data) / len(data) * sub_entropy
    return init_entropy - new_entropy


def shuffle_dataset(dataset: pd.DataFrame):
    """
    Shuffles the dataset and the target labels.
    Arguments:
        data: A numpy array of shape (n_samples, n_features)
        targets: A numpy array of shape (n_samples, )
    Returns:
        A shuffled dataset and target labels.
    """
    dataset = dataset.sample(frac=1, replace=False)
    return dataset


def split_dataset(
    data: pd.DataFrame,
    targets: pd.Series,
    split_size: float = 0.2
) -> Tuple[Tuple[pd.DataFrame, pd.Series], Tuple[pd.DataFrame, pd.Series]]:
    num_samples = data.shape[0]
    num_train = int(num_samples*split_size)

    train_data = data.iloc[:num_train]
    train_targets = targets.iloc[:num_train]

    valid_data = data.iloc[num_train:]
    valid_targets = targets.iloc[num_train:]

    return (train_data, train_targets), (valid_data, valid_targets)


def get_metrics(
    y_pred=None,
    y_true=None,
    metrics: List[str] = ['Accuracy'],
    classes: List[str] = None,
) -> None:

    if isinstance(y_pred, np.ndarray) == False:
        y_pred = y_pred.to_numpy()
    if isinstance(y_true, np.ndarray) == False:
        y_true = y_true.to_numpy()

    results = {}
    for metric in metrics:
        if metric == 'Accuracy':
            results[metric] = accuracy(y_true, y_pred)
        elif metric == 'Precision':
            results[metric] = []
            for i, label in enumerate(classes):
                results[metric].append(precision_score(
                    y_true, y_pred, label=i))
        elif metric == 'Recall':
            results[metric] = []
            for i, label in enumerate(classes):
                results[metric].append(recall_score(
                    y_true, y_pred, label=i))
        elif metric == 'F1':
            results[metric] = []
            for i, label in enumerate(classes):
                results[metric].append(f1_score(
                    y_true, y_pred, label=i))
        elif metric == 'Confusion Matrix':
            results[metric] = confusion_matrix(
                y_true, y_pred, labels=classes)
        else:
            raise ValueError('Unknown metric: {}'.format(metric))
    return results


def accuracy(y_true, y_pred):
    return np.mean(y_true == y_pred)


def precision_score(y_true, y_pred, label: int):
    true_pos = np.sum(np.logical_and(y_true == label, y_pred == label))
    return true_pos / (np.sum(y_pred == label) + 1e-7)


def recall_score(y_true, y_pred, label: int):
    true_pos = np.sum(np.logical_and(y_true == label, y_pred == label))
    return true_pos / np.sum(y_true == label)


def f1_score(y_true, y_pred, label: int):
    precision = precision_score(y_true, y_pred, label)
    recall = recall_score(y_true, y_pred, label)
    return 2 * (precision * recall) / (precision + recall + 1e-7)


def confusion_matrix(y_true, y_pred, labels: List = []):
    matrix = np.zeros((len(labels), len(labels)))
    for label, label_name in enumerate(labels):
        true_pos = np.sum(np.logical_and(y_true == label, y_pred == label))
        matrix[label, label] = true_pos
        for i, other_label in enumerate(labels):
            if i != label:
                matrix[label, i] = np.sum(
                    np.logical_and(y_true == label, y_pred == i))
    return matrix
