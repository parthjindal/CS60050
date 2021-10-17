from typing import Callable
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from util import Vectorizer, MailDataset, get_euclidean_distance


class KNN:
    def __init__(
        self,
        x_train: np.ndarray,
        y_train: np.ndarray,
        criteria: Callable,
        k: int,
    ) -> None:

        self.x_train = x_train
        self.y_train = y_train
        self.criteria = criteria
        self.k = k

    def _predict_one(self, x_test: np.ndarray) -> np.ndarray:
        x_test = x_test.reshape(1, -1)
        distances = self.criteria(x_test, self.x_train)
        min_indices = np.argsort(distances, axis=1)[:, :self.k]
        predictions = self.y_train[min_indices]
        predict = np.apply_along_axis(
            lambda x: np.bincount(x).argmax(), axis=1, arr=predictions
        )
        return predict

    def predict(self, X: np.ndarray) -> np.ndarray:
        predicts = []
        for x in X:
            predicts.append(self._predict_one(x))
        return predicts


def test():
    dataset = MailDataset(root='./dataset')
    x_train, y_train = dataset.X, dataset.Y
    vectorizer = Vectorizer()
    x_train = vectorizer.fit_transform(x_train).to_numpy().astype(np.float32)
    y_train = y_train.to_numpy().astype(int)
    knn = KNN(x_train=x_train,
              y_train=y_train,
              criteria=get_euclidean_distance,
              k=1)
    predicts = knn.predict(x_train[10:40])
    print(f'Accuracy: {np.mean(predicts == y_train[10:40])}')


if __name__ == '__main__':
    test()
