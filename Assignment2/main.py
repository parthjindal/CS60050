from util import seed_everything, split_dataset,\
    shuffle_dataset, Vectorizer,\
    get_cosine_score, get_euclidean_distance, \
    get_manhattan_distance, MailDataset, get_metrics
from knn import KNN
import matplotlib.pyplot as plt
import argparse
import numpy as np
from tabulate import tabulate


CRITERION = {
    "euclidean": get_euclidean_distance,
    "manhattan": get_manhattan_distance,
    "cosine": get_cosine_score
}


def main(args: argparse.Namespace):
    vectorizer = Vectorizer(max_features=args.max_features)
    dataset = MailDataset(root="./dataset", transform=vectorizer)
    shuffle_dataset(dataset)

    X, Y = dataset.X, dataset.Y
    (x_train, y_train), (x_test, y_test) = split_dataset(X, Y, split_size=0.2)

    knn = KNN(x_train, y_train, criteria=CRITERION[args.criteria], k=args.k)
    y_pred = knn.predict(x_test)
    print(f"Accuracy: {np.mean(y_pred == y_test)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="KNN")
    parser.add_argument("--k", type=int, default=5, help="Number of neighbors")
    parser.add_argument("--criteria", type=str, default="euclidean",
                        help="Distance metric", choices=["cosine", "euclidean", "manhattan"])
    parser.add_argument("--max_features", type=int, default=1000,
                        help="Number of features to use in vectorization (TF-IDF")
    parser.add_argument("-v", "--verbose",
                        action="store_true", help="Verbose mode")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    seed_everything(args.seed)

    main(args)
