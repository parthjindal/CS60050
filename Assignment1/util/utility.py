from typing import List, Tuple
import numpy as np
import pandas as pd
from scipy import stats
import random


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)


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


def shuffle_dataset(dataset):
    """
    Shuffles the dataset and the target labels.
    Arguments:
        data: A numpy array of shape (n_samples, n_features)
        targets: A numpy array of shape (n_samples, )
    Returns:
        A shuffled dataset and target labels.
    """
    dataset.df = dataset.df.sample(frac=1, replace=False)


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
