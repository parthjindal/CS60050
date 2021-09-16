from util import CarDataset
from decision_tree import DecisionTree
import numpy as np
import matplotlib.pyplot as plt
import random
import argparse


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
 

def main(args: argparse.Namespace):
    dataset = CarDataset("./dataset", download=True)

    x_train, y_train = dataset.data, dataset.targets
    permuts = np.random.permutation(len(dataset))
    x_train, y_train = x_train[permuts], y_train[permuts]

    dt = DecisionTree(max_depth=args.max_depth,
                      min_samples_leaf=args.min_samples_leaf,
                      criterion=args.criterion)
    ...


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Decision Tree')
    parser.add_argument('--max_depth', type=int, default=5,
                        help='Maximum depth of the tree')
    parser.add_argument('--min_samples', type=int, default=5,
                        help='Minimum number of samples in a leaf')
    parser.add_argument('--criterion', type=str, default='gini',
                        help='Split criteria', choices=['gini', 'entropy'])
    parser.add_argument('--seed', type=int, default=42)

    args = parser.parse_args()

    seed_everything(args.seed)
    main(args)
