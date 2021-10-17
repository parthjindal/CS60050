import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple
import random
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


def get_cosine_similarity(x: np.ndarray, y: np.ndarray):
    """
    Computes the cosine similarity between two vectors.
    Args:
        x: A numpy array of shape (x_samples, n_features)
        y: A numpy array of shape (y_samples, n_features)
    Returns:
        A numpy array of shape (x_samples, y_samples)
    """
    return np.dot(x, y.T) / \
        (np.linalg.norm(x, axis=1, keepdims=True) *
         np.linalg.norm(y.T, axis=0, keepdims=True))


def get_euclidean_distance(x: np.ndarray, y: np.ndarray):
    """
    Computes the euclidean distance between two vectors.
    Args:
        x: A numpy array of shape (x_samples, n_features)
        y: A numpy array of shape (y_samples, n_features)
    Returns:
        A numpy array of shape (x_samples, y_samples)
    """
    x = x[..., np.newaxis]
    y = y[..., np.newaxis]
    return np.linalg.norm(x - y.T, axis=1, ord=2)


def get_manhattan_distance(x: np.ndarray, y: np.ndarray):
    """
    Computes the manhattan distance between two vectors.
    Args:
        x: A numpy array of shape (x_samples, n_features)
        y: A numpy array of shape (y_samples, n_features)
    Returns:
        A numpy array of shape (x_samples, y_samples)
    """
    x = x[..., np.newaxis]
    y = y[..., np.newaxis]
    return np.linalg.norm(x - y.T, axis=1, ord=1)


class Vectorizer:
    def __init__(
        self,
        max_features: int = 1000,
        min_df: int = 1,
        max_df: float = 0.9,
        ngram_range: tuple = (1, 1),
    ) -> None:

        self.vectorizer = TfidfVectorizer(
            max_features=max_features,
            min_df=min_df,
            max_df=max_df,
            ngram_range=ngram_range,
        )

    def fit(self, df: pd.DataFrame) -> None:
        self.vectorizer.fit(df.text)

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        return pd.DataFrame(
            self.vectorizer.transform(df.text).toarray(),
            columns=self.vectorizer.get_feature_names(),
        )

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        self.fit(df)
        return self.transform(df)

        ...
    __call__ = fit_transform


def test():
    vectorizer = Vectorizer()
    dataset = MailDataset(root="../dataset")
    df = dataset.df
    vectorizer.fit(df)
    df_vectorized = vectorizer(df)
    print(df_vectorized.shape)


# test()
