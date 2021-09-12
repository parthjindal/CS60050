import numpy as np
import matplotlib.pyplot as plt
import graphviz
import pandas as pd


def gini_index(targets):
    label_count = np.bincount(targets)
    return 1 - np.sum((label_count / len(targets)) ** 2)


def entropy(targets):
    label_count = np.bincount(targets)
    return -np.sum(label_count / len(targets) * np.log2(label_count / len(targets)))


def gini_gain(data: pd.DataFrame, attribute: str, targets: pd.Series) -> float:
    """
    Computes the gini gain for a given attribute and target labels.
    Arguments:
        data: A numpy array of shape (n_samples, n_features)
        attribute: The index of the attribute to compute the gini gain for
        targets: A numpy array of shape (n_samples, )
    Returns:
        The gini gain for the given attribute and target labels.
    """
    init_gini = gini_index(targets)  # Initial gini index
    # All possible values of the attribute
    attribute_values = np.unique(data[:, attribute])
    new_gini = 0  # Initialize the new gini index

    for value in attribute_values:  # For each value of the attribute
        sub_data = data[data[:, attribute] == value]
        sub_targets = targets[data[:, attribute] == value]
        sub_gini = gini_index(sub_targets)
        new_gini += len(sub_data) / len(data) * sub_gini
    return init_gini - new_gini  # Return the gini gain


def information_gain(data: pd.DataFrame, attribute: str, targets: pd.Series):
    """
    Computes the information gain for a given attribute and target labels.
    Arguments:
        data: A numpy array of shape (n_samples, n_features)
        attribute: The index of the attribute to compute the information gain for
        targets: A numpy array of shape (n_samples, )
    Returns:
        The information gain for the given attribute and target labels.
    """
    init_entropy = entropy(targets)
    attribute_values = np.unique(data[:, attribute])
    new_entropy = 0
    for value in attribute_values:
        sub_data = data[data[:, attribute] == value]
        sub_targets = targets[data[:, attribute] == value]
        sub_entropy = entropy(sub_targets)
        new_entropy += len(sub_data) / len(data) * sub_entropy
    return init_entropy - new_entropy
