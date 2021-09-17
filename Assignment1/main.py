from posixpath import split
from typing import List, Tuple
import pandas as pd
from util import CarDataset, dataset, split_dataset, shuffle_dataset, get_metrics
from decision_tree import DecisionTree
import numpy as np
import matplotlib.pyplot as plt
import random
import argparse
from tabulate import tabulate


def train(
    dataset: CarDataset,
    args: argparse.Namespace,
    metrics: List[str],
    **kwargs
):
    class_names = dataset.metadata['class']
    x, y = dataset.data, dataset.targets
    (x_train, y_train), (x_test, y_test) = split_dataset(x, y, 0.8)

    dt = DecisionTree(max_depth=args.max_depth,
                      min_samples_leaf=args.min_samples,
                      criterion=args.criterion)
    dt.fit(x_train, y_train)

    y_pred = dt.predict(x_test)
    results = get_metrics(y_test, y_pred, metrics, classes=class_names)

    return results


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)


def process_data(data):
    print(data.value_counts())


def main(args):
    dataset = CarDataset(root="./dataset", download=False)
    dataset.df = shuffle_dataset(dataset.df)
    class_names = dataset.metadata['class']
    results = train(dataset, args, ['Accuracy', 'Precision',
                                    'Recall', 'F1', 'Confusion Matrix'], verbose=True)

    if args.verbose:
        print("Test Accuracy: ", results['Accuracy'])
        print("\nStatistics:")
        print_dict = {
            'Class': class_names,
            'Precision': results['Precision'],
            'Recall': results['Recall'],
            'F1': results['F1']
        }
        print(tabulate(print_dict, headers='keys', tablefmt='psql'))

        print("\nConfusion Matrix:")
        cnf_matrix = {
            'Predictions: ': class_names,
        }
        for i in range(len(class_names)):
            cnf_matrix[class_names[i]] = results['Confusion Matrix'][i]
        print(tabulate(cnf_matrix, headers='keys', tablefmt='psql'))


if __name__ == '__main__':
    # process_data()
    parser = argparse.ArgumentParser(description='Decision Tree')
    parser.add_argument('--max_depth', type=int, default=6,
                        help='Maximum depth of the tree')
    parser.add_argument('--min_samples', type=int, default=1,
                        help='Minimum number of samples in a leaf')
    parser.add_argument('--criterion', type=str, default='gini',
                        help='Split criteria', choices=['gini', 'entropy'])
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--verbose', action='store_true')

    args = parser.parse_args()

    seed_everything(args.seed)
    main(args)
