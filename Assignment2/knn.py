import numpy as np
from util import get_cosine_score, minkowski_distance
from typing import Callable, Tuple, Union


class KNN:
    def __init__(
        self,
        x_train: np.ndarray,
        y_train: np.ndarray,
        n_neighbours: int,
        weighted: bool = False,
        metric: Union[str, Callable] = "euclidean",
        method: str = "brute",
        dst_mtrx: np.ndarray = None,
    ) -> None:
        """
        Initialize KNN classifier

        Args:
            x_train : ndarray of shape (n_samples, n_features)
            y_train : ndarray of shape (n_samples,)
            criteria : Either a string from ['euclidean', 'cosine', 'manhattan'] or a user defined callable
            n_neighbors : Number of neighbors to use for prediction
            weighted : Whether to use weighted voting or not (Default: False)
            method : Either 'brute' or 'cache' (Default: 'brute')
        """

        self.x_train = x_train
        self.y_train = y_train
        self.criteria: Callable = None
        self.n_neighbours = n_neighbours
        self.weighted = weighted
        self._fnkwargs = {}
        self.metric = None
        self.method = method
        self._dst_mtrx = dst_mtrx

        if isinstance(metric, str):
            self.metric = metric
            if metric == "cosine":
                self.criteria = get_cosine_score
            elif metric == "euclidean":
                self.criteria = minkowski_distance
                self._fnkwargs["p"] = 2
            elif metric == "manhattan":
                self.criteria = minkowski_distance
                self._fnkwargs["p"] = 1
            else:
                raise ValueError("Invalid criteria for KnearestNeighbour")
        elif isinstance(metric, Callable):
            self.criteria = metric
            self.metric = "custom"
        else:
            raise TypeError("Criteria must be str | Callable")

    def _compute_chunked_distance(self, X: np.ndarray) -> np.ndarray:
        CHUNK_SIZE = 1000000 // (self.x_train.shape[0]
                                 * self.x_train.shape[1]) + 1
        slices = np.array_split(X, X.shape[0] // CHUNK_SIZE + 1)
        distances = np.zeros((X.shape[0], self.x_train.shape[0]))
        index = 0
        for chunk in slices:
            width = chunk.shape[0]
            distances[index: index + width] = self.criteria(
                chunk, self.x_train, **self._fnkwargs)
            index += width
        return distances

    def neighbours(self, X: np.ndarray, n_neighbours=None) -> Tuple[np.ndarray, np.ndarray]:
        if n_neighbours is None:
            n_neighbours = self.n_neighbours
        if X.ndim == 1:
            X = X.reshape((1, -1))
        if self.metric == "cosine":
            distances = self.criteria(X, self.x_train, **self._fnkwargs)
        elif self.metric in ["euclidean", "manhattan"]:
            distances = self._compute_chunked_distance(X)
        else:
            distances = self.criteria(X, self.x_train, **self._fnkwargs)
        min_indices = np.argsort(distances, axis=1)[:, :n_neighbours]
        return distances, min_indices

    def _predict_one(self, x: np.ndarray, distances, min_indices, weighted=False) -> np.ndarray:
        votes = self.y_train[min_indices]
        if weighted:
            weights = 1 / (distances ** 2 + 1e-8)
            return np.argmax(np.bincount(votes, weights=weights))
        else:
            return np.argmax(np.bincount(votes))

    def predict(self, X: np.ndarray) -> np.ndarray:
        if X.ndim == 1:
            X = X.reshape((1, -1))
        predicts = np.zeros(X.shape[0], dtype=np.int32)
        distances = np.zeros((X.shape[0], self.x_train.shape[0]))
        min_indices = np.zeros((X.shape[0], self.n_neighbours), dtype=np.int32)

        if self.method == "brute":
            distances[...], min_indices[...] = self.neighbours(X)
        elif self.method == "cached":
            distances[...] = self._dst_mtrx
            min_indices[...] = np.argsort(distances, axis=1)[
                :, :self.n_neighbours]
        votes = self.y_train[min_indices]

        for i in range(X.shape[0]):
            predicts[i] = self._predict_one(
                X[i], distances[i, min_indices[i]], min_indices[i], self.weighted)
        return predicts


def test():
    from util import shuffle_dataset
    from util import MailDataset, Vectorizer
    # import KnearestNeighbour from sklearn
    from sklearn.neighbors import KNeighborsClassifier
    vectorizer = Vectorizer(1000, 1, 1.0)

    dataset = MailDataset(root='./dataset', transform=vectorizer)
    shuffle_dataset(dataset)

    X, Y = dataset.X, dataset.Y
    x_train = X[:int(0.8 * len(X))]
    y_train = Y[:int(0.8 * len(X))]

    knn = KNN(x_train=x_train,
              y_train=y_train,
              metric="cosine",
              n_neighbours=11, weighted=False)
    OG_knn = KNeighborsClassifier(n_neighbors=51, metric="cosine")
    OG_knn.fit(x_train, y_train)
    predicts = knn.predict(X[int(0.8 * len(X)):])
    predicts_2 = OG_knn.predict(X[int(0.8 * len(X)):])
    print(f'Accuracy: {np.mean(predicts == Y[int(0.8 * len(X)):])}')
    print(f'Accuracy: {np.mean(predicts_2 == Y[int(0.8 * len(X)):])}')


if __name__ == '__main__':
    test()
