from typing import Callable
import numpy as np


class KNN:

    def __init__(
        self,
        x_train: np.ndarray,
        y_train: np.array,
        criteria: Callable,
        k: int,
    ) -> None:

        self.x_train = x_train
        self.y_train = y_train

        self.criteria = criteria
        self.k = k

    def _predict_one(self, x_test: np.ndarray) -> np.ndarray:  # hack for mem-runout
        x_test = x_test.reshape((1, -1))
        distances = self.criteria(x_test, self.x_train)
        min_indices = np.argsort(distances, axis=1)[:, :self.k]
        predictions = self.y_train[min_indices]
        predict = np.apply_along_axis(
            lambda x: np.bincount(x).argmax(), axis=1, arr=predictions
        )
        return predict

    def predict(self, X: np.ndarray) -> np.ndarray:
        if X.ndim == 1:
            X = X.reshape((1, -1))
        predicts = np.zeros(X.shape[0], dtype=np.int32)
        for i in range(X.shape[0]):
            predicts[i] = self._predict_one(X[i])
        return predicts


def test():
    from util import shuffle_dataset, split_dataset
    from util import Vectorizer, MailDataset, get_euclidean_distance, get_manhattan_distance, get_cosine_score

    vectorizer = Vectorizer(500)
    dataset = MailDataset(root='./dataset', transform=vectorizer)
    shuffle_dataset(dataset)

    X, Y = dataset.X, dataset.Y
    x_train = X[:int(0.8 * len(X))]
    y_train = Y[:int(0.8 * len(X))]

    knn = KNN(x_train=x_train,
              y_train=y_train,
              criteria=get_cosine_score,
              k=11)
    predicts = knn.predict(X[int(0.8 * len(X)):])
    print(f'Accuracy: {np.mean(predicts == Y[int(0.8 * len(X)):])}')


if __name__ == '__main__':
    test()
