import random
import numpy as np
from typing import Tuple
from sklearn.feature_extraction.text import TfidfVectorizer


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)


def shuffle_dataset(dataset):
    """
    Shuffles the dataset and the target labels.
    Arguments:
        data: A numpy array of shape (n_samples, n_features)
        targets: A numpy array of shape (n_samples, )
    Returns:
        A shuffled dataset and target labels.
    """
    permute = np.random.permutation(len(dataset))
    dataset.X = dataset.X[permute]
    dataset.Y = dataset.Y[permute]


def split_dataset(
    data: np.ndarray,
    targets: np.ndarray,
    split_size: float = 0.2
) -> Tuple[Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray]]:
    """
    Splits the dataset into training and test set.
    Args:
        data :Numpy ndarray containing the data.
        targets : Target label numpy ndarray.
        split_size : % data assigned to test set. Defaults to 0.2.

    Returns:
        Tuple containing the training and test set.
    """
    num_samples = data.shape[0]
    num_train = int(num_samples*(1-split_size))

    train_data = data[:num_train]
    train_targets = targets[:num_train]

    valid_data = data[num_train:]
    valid_targets = targets[num_train:]

    return (train_data, train_targets), (valid_data, valid_targets)


def get_cosine_score(x: np.ndarray, y: np.ndarray, *args, **kwargs) -> np.ndarray:
    """
    Computes the cosine score between two vectors.
    Args:
        x: A numpy array of shape (x_samples, n_features)
        y: A numpy array of shape (y_samples, n_features)
    Returns:
        A numpy array of shape (x_samples, y_samples)
    """
    return -((x @ y.T) /
             (np.linalg.norm(x, axis=1, keepdims=True, ord=2) *
              np.linalg.norm(y.T, axis=0, keepdims=True, ord=2) + 1e-8))


def minkowski_distance(x: np.ndarray, y: np.ndarray, p=2, *args, **kwargs) -> np.ndarray:
    """
    Computes the pth order minkowski distance between two vectors.
    Args:
       x: A numpy array of shape (x_samples, n_features)
       y: A numpy array of shape (y_samples, n_features)
    Returns:
       A numpy array of shape (x_samples, y_samples)
    """
    x = x[..., np.newaxis]
    y = y[..., np.newaxis]
    # use with caution: may run out of memory (super-fast though xD)
    # todo: add a (m * n * k) <= threshold and if exceeds do simple iter
    return np.linalg.norm(x - y.T, axis=1, ord=p)


class Vectorizer:
    """
    Wrapper vectorizer class
    """

    def __init__(
        self,
        max_features: int = None,
        min_df: int = 1,
        max_df: float = 0.9,
        ngram_range: tuple = (1, 1),
    ) -> None:

        self.vectorizer = TfidfVectorizer(
            max_features=max_features,
            min_df=min_df,
            max_df=max_df,
            ngram_range=ngram_range,
            stop_words='english',
        )

    def fit_transform(self, data: np.ndarray) -> np.ndarray:
        return self.vectorizer.fit_transform(data).toarray()

    def transform(self, data: np.ndarray) -> np.ndarray:
        return self.vectorizer.transform(data).toarray()

    __call__ = fit_transform
