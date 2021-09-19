from typing import List, Dict, Tuple
from util import CarDataset, split_dataset, shuffle_dataset, \
    get_metrics, find_ci_interval, seed_everything
from decision_tree import DecisionTree
import numpy as np
import matplotlib.pyplot as plt
import argparse
from tabulate import tabulate


def train(
    dataset: CarDataset,
    max_depth=None,
    criterion='gini',
    min_samples=1,
    metrics: List[str] = [],
    **kwargs
):
    class_names = dataset.metadata['class']
    x, y = dataset.data, dataset.targets
    (x_train, y_train), (x_test, y_test) = split_dataset(x, y, 0.8)

    dt = DecisionTree(max_depth, min_samples, criterion)
    dt.fit(x_train, y_train)

    y_pred = dt.predict(x_test)
    results = get_metrics(y_test, y_pred, metrics, classes=class_names)

    return results, dt


def compare_criteria(
    num_runs: int = 1,
    args: argparse.Namespace = None
):
    """
    Wrapper function to compare different criteria for the Decision Tree.
    Arguments:
        num_runs: Number of times to run the experiment.
                  Accuracy is averaged across all runs and a 95% confidence interval is computed.
        args: Arguments to pass to the train function.
    """

    dataset = CarDataset(root="./dataset", download=False)
    criteria = ['gini', 'ig']
    metrics = ['Accuracy']

    results = {'gini': [], 'ig': []}
    for criterion in criteria:
        for _ in range(num_runs):
            shuffle_dataset(dataset)
            result, _ = train(dataset,
                              args.max_depth,
                              criterion,
                              args.min_samples,
                              metrics)
            results[criterion].append(result['Accuracy'])

    gini_results = results['gini']
    entropy_results = results['ig']
    print_dict = {
        'Criterion': ["Gini Index", "Entropy"],
        'Mean Accuracy': [],
        '95% CI': []
    }

    mean, ci = find_ci_interval(gini_results, confidence=0.95)
    print_dict['Mean Accuracy'].append(f"{mean*100:.2f}%")
    print_dict['95% CI'].append(f"{ci[0]*100:.2f}% - {ci[1]*100:.2f}%")

    mean, ci = find_ci_interval(entropy_results, confidence=0.95)
    print_dict['Mean Accuracy'].append(f"{mean*100:.2f}%")
    print_dict['95% CI'].append(f"{ci[0]*100:.2f}% - {ci[1]*100:.2f}%")

    print(tabulate(print_dict, headers='keys',
                   tablefmt='psql', floatfmt=".2f",))


def average_runs(
    args,
    num_runs=10,
    verbose=False
):
    metrics = ['Accuracy', 'Precision', 'Recall', 'F1', 'Confusion Matrix']

    best_result: Dict = None
    best_dataset: CarDataset = None
    best_dt: DecisionTree = None
    best_acc: float = 0.0
    accs = []

    for i in range(num_runs):
        dataset = CarDataset(root="./dataset", download=False)
        shuffle_dataset(dataset)

        result, tree = train(dataset,
                             args.max_depth,
                             args.criterion,
                             args.min_samples,
                             metrics)

        accs.append(result['Accuracy'])
        if accs[-1] > best_acc:
            best_acc = accs[-1]
            best_result = result
            best_dataset = dataset
            best_dt = tree

    mean_acc, ci = find_ci_interval(accs, confidence=0.95)

    verbose = True
    if verbose:
        class_names = best_dataset.metadata['class']
        print(f"\nMean Accuracy: {mean_acc*100:.2f}%")
        print('-'*25)
        print(f"95% CI: {ci[0]*100:.2f}% - {ci[1]*100:.2f}%")
        print("\nBest Run Statistics:")
        print("Accuracy: ", '%.2f' % (best_acc*100), "%")

        stats = {
            'Class': class_names,
            'Precision': best_result['Precision'],
            'Recall': best_result['Recall'],
            'F1': best_result['F1']
        }
        print(tabulate(stats, headers='keys', tablefmt='psql', floatfmt=".3f"))

        print("\nConfusion Matrix:")
        cnf_matrix = {'Predictions: ': class_names}
        for i in range(len(class_names)):
            cnf_matrix[class_names[i]] = best_result['Confusion Matrix'][i]

        print(tabulate(cnf_matrix, headers='keys', tablefmt='psql'))
        best_dt.print_tree(best_dataset.metadata)
    return best_dt, best_dataset, best_result


def compare_heights(
    args,
    num_runs=10,
):
    heights = np.arange(0, 8, dtype=int)
    results = np.zeros((num_runs, len(heights)))
    num_nodes = {}

    dataset = CarDataset(root="./dataset", download=False)

    for run in range(num_runs):
        shuffle_dataset(dataset)
        for height in heights:
            result, dt = train(dataset, max_depth=height,
                               min_samples=1, criterion='ig', metrics=['Accuracy'])
            results[run][height] = result['Accuracy']
            num_nodes[dt.num_nodes] = result['Accuracy']
    # plot the height vs accuracy with error bars
    means, ci = find_ci_interval(results, confidence=0.95)
    h = ci[1] - means

    # plot the height vs accuracy with error bars (95% CI) with circles at the mean and connecting lines
    plt.errorbar(heights, means, yerr=h, fmt='--o',
                 dash_capstyle='round', capsize=5)
    plt.xlabel("Max Tree Height")
    plt.ylabel("Accuracy")
    plt.show()

    num_nodes = sorted(num_nodes.items(), key=lambda x: x[0])

    # plot the number of nodes vs accuracy
    plt.plot([x[0] for x in num_nodes], [x[1] for x in num_nodes])
    plt.xlabel("Number of Nodes")
    plt.ylabel("Accuracy")
    plt.show()


def prune_tree(dt: DecisionTree, x_valid, y_valid, class_names):
    """
    Prunes the tree using the validation set.
    Arguments:
        dt: Decision Tree to be pruned.
        dataset: Validation dataset.
    """
    metrics = ['Accuracy', 'Precision', 'Recall', 'F1', 'Confusion Matrix']
    y_preds = dt.predict(x_valid)
    results = get_metrics(y_valid, y_preds, metrics, class_names)
    print("\n--------------------Before Pruning Statistics-----------------")

    print(f"\nValidation Set Metrics:")
    print("Accuracy: ", '%.2f' % (results['Accuracy']*100), "%")
    print("Number of Nodes: {}".format(dt.num_nodes))

    stats = {
        'Class': class_names,
        'Precision': results['Precision'],
        'Recall': results['Recall'],
        'F1': results['F1']
    }
    print(tabulate(stats, headers='keys', tablefmt='psql', floatfmt=".3f"))

    print("\nConfusion Matrix:")
    cnf_matrix = {'Predictions: ': class_names}
    for i in range(len(class_names)):
        cnf_matrix[class_names[i]] = results['Confusion Matrix'][i]

    print(tabulate(cnf_matrix, headers='keys', tablefmt='psql'))

    print("\n--------------------After Pruning Statistics-----------------")
    dt.prune_tree('reduced-error', x_valid, y_valid)

    y_preds = dt.predict(x_valid)
    results = get_metrics(y_valid, y_preds, metrics, class_names)

    print(f"\nValidation Set Metrics:")
    print("Accuracy: ", '%.2f' % (results['Accuracy']*100), "%")
    print("Number of Nodes: {}".format(dt.num_nodes))

    stats = {
        'Class': class_names,
        'Precision': results['Precision'],
        'Recall': results['Recall'],
        'F1': results['F1']
    }
    print(tabulate(stats, headers='keys', tablefmt='psql', floatfmt=".3f"))

    print("\nConfusion Matrix:")
    cnf_matrix = {'Predictions: ': class_names}
    for i in range(len(class_names)):
        cnf_matrix[class_names[i]] = results['Confusion Matrix'][i]

    print(tabulate(cnf_matrix, headers='keys', tablefmt='psql'))


def main(args):
    dataset = CarDataset(root="./dataset", download=False)
    dataset.df = shuffle_dataset(dataset)
    class_names = dataset.metadata['class']
    results = train(dataset, args, ['Accuracy', 'Precision',
                                    'Recall', 'F1', 'Confusion Matrix'], verbose=True)

    if args.verbose:
        print("Test Accuracy: ", '%.2f' % (results['Accuracy']*100))
        print("\nStatistics:")
        print_dict = {
            'Class': class_names,
            'Precision': results['Precision'],
            'Recall': results['Recall'],
            'F1': results['F1']
        }
        print(tabulate(print_dict, headers='keys',
              tablefmt='psql', floatfmt=".3f"))

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
                        help='Split criteria', choices=['gini', 'ig'])
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--verbose', action='store_true')

    args = parser.parse_args()

    seed_everything(args.seed)
    # compare_criteria(num_runs=10, args=args)
    dt, dataset, _ = average_runs(args, 10, args.verbose)
    # (x_train, y_train), (x_valid, y_valid) = split_dataset(
    #     dataset.data, dataset.targets, 0.8)
    # prune_tree(dt, x_valid, y_valid, dataset.metadata['class'])

    # compare_heights(args, num_runs=10)
