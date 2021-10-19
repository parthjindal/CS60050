from re import M
from util import seed_everything, split_dataset,\
    shuffle_dataset, Vectorizer, MailDataset, get_metrics, accuracy
from knn import KNN
import argparse
import numpy as np
from tabulate import tabulate
import matplotlib.pyplot as plt


def main(args: argparse.Namespace):
    # vectorizer (TF_IDF)
    vectorizer = Vectorizer(max_features=args.max_features,
                            min_df=args.min_df,
                            max_df=args.max_df)

    dataset = MailDataset(root="./dataset")
    shuffle_dataset(dataset)
    (x_train, y_train), (x_test, y_test) = split_dataset(
        dataset.X, dataset.Y, split_size=0.2)

    x_train = vectorizer.fit_transform(x_train)  # vectorize training data
    x_test = vectorizer.transform(x_test)  # transform test data

    knn = KNN(x_train, y_train, metric=args.metric,
              n_neighbours=args.n, weighted=False, method="brute")
    y_pred = knn.predict(x_test)
    metrics = ["Accuracy", "Precision", "Recall", "F1", "Confusion Matrix"]
    results = get_metrics(y_pred, y_test, metrics)

    print(f"Accuracy: {results['Accuracy']*100:.2f}%")
    print(f"Precision: {results['Precision']*100:.2f}%")
    print(f"Recall: {results['Recall']*100:.2f}%")
    print(f"F1: {results['F1']*100:.2f}%")

    cnf_matrix = {
        "True Label": ["ham", "spam"],
    }
    for i, col in enumerate(results["Confusion Matrix"].T):
        cnf_matrix[f"Predicted {i}"] = col
    print(tabulate(cnf_matrix, headers="keys"))


def compare_metrics(args, metrics=["cosine", "euclidean", "manhattan"]):

    vectorizer = Vectorizer(max_features=args.max_features,
                            min_df=args.min_df,
                            max_df=args.max_df)

    dataset = MailDataset(root="./dataset")
    shuffle_dataset(dataset)
    (x_train, y_train), (x_test, y_test) = split_dataset(
        dataset.X, dataset.Y, split_size=0.2)

    x_train = vectorizer.fit_transform(x_train)  # vectorize training data
    x_test = vectorizer.transform(x_test)  # transform test data

    results = {}
    K_range = np.arange(1, 200)

    for metric in metrics:
        results[metric] = []
        # precompute distance matrices,sorted-indices for all metrics
        knn = KNN(x_train, y_train, metric=metric, n_neighbours=len(
            x_train), weighted=False, method="brute")
        dst_mtrx, min_indices = knn.neighbours(x_test)

        for k in K_range:
            predicts = np.zeros(len(x_test), dtype=int)
            for i in range(len(x_test)):
                kmin_indices = min_indices[i][:k]
                kdist_mtrx = dst_mtrx[i][kmin_indices]
                predicts[i] = knn._predict_one(
                    x_test[i], kdist_mtrx, kmin_indices, weighted=False)
            acc = accuracy(y_test, predicts)
            print(f"{metric} K={k}: {acc*100:.2f}%")
            results[metric].append(acc)

    # plot result for each metric and save figures for each metric
    for metric in metrics:
        fig = plt.figure()
        plt.plot(K_range, results[metric], label=metric)
        plt.xlabel("K")
        plt.ylabel("Accuracy")
        plt.title(f"Accuracy vs K for {metric}")
        plt.legend()
        fig.savefig(f"{metric}.png")

    print("Comparing metrics completed!")


def average_runs(args, num_runs=10):
    # TF-IDF vectorizer used
    vectorizer = Vectorizer(max_features=args.max_features,
                            min_df=args.min_df,
                            max_df=args.max_df)
    dataset = MailDataset(root="./dataset", transform=vectorizer)
    results = []
    for i in range(num_runs):
        shuffle_dataset(dataset)
        (x_train, y_train), (x_test, y_test) = split_dataset(
            dataset.X, dataset.Y, split_size=0.2)
        knn = KNN(x_train, y_train, metric="", n_neighbours=args.n)
        y_pred = knn.predict(x_test)
        results.append(get_metrics(y_test, y_pred, [
                       "Accuracy", "Precision", "Recall", "F1"]))
    print(tabulate(results, headers=["Accuracy", "Precision", "Recall", "F1"]))


def compare_measures(args, measures=["euclidean", "manhattan", "cosine"]):
    ...


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="KNN")
    parser.add_argument("--n", type=int, default=5, help="Number of neighbors")
    parser.add_argument("--metric", type=str, default="euclidean",
                        help="Distance metric", choices=["cosine", "euclidean", "manhattan"])
    parser.add_argument("--weighted", type=bool, default=False,
                        help="Weighted KNN")
    parser.add_argument("--max_features", type=int, default=1000,
                        help="Number of features to use in vectorization (TF-IDF")
    parser.add_argument("--min_df", type=int, default=1,
                        help="Minimum document frequency")
    parser.add_argument("--max_df", type=float, default=1.0,
                        help="Maximum document frequency")
    parser.add_argument("-v", "--verbose",
                        action="store_true", help="Verbose mode")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    seed_everything(args.seed)

    print("---------Results on 80/20 Split---------")
    # main(args)
    compare_metrics(args, metrics=["manhattan", "cosine", "euclidean"])
