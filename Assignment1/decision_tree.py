from util import shuffle_dataset, split_dataset, chi2_contingency
import pandas as pd
from util import information_gain, gini_gain
import graphviz
from typing import List, Tuple, Dict
from util import CarDataset
import numpy as np



class DecisionTree():
    """
    Decision Tree model for supervised learning
    Args:
        max_depth (int): Maximum depth of any branch in the Decision tree.Defaults to 10.
        min_samples_leaf (int): Minimum no. of samples for further splitting nodes else treated as a leave. Defaults to 1.
        criterion (str): Criteria to select the attribute at a node(['gini', 'ig']). Defaults to "gini".
    """

    def __init__(
        self,
        max_depth: int = 10,
        min_samples_leaf: int = 1,
        criterion: str = "gini"
    ) -> None:
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.criterion = criterion
        self.tree = None
        self.final_depth = 0
        self.num_nodes = 0
        self.num_classes = 4

    def fit(self, X: pd.DataFrame, y: pd.Series) -> None:
        """
        This function fits the model to the data.
        Args:
            X (pd.DataFrame): Input dataset as a pandas DataFrame.
            y (pd.Series):    Target values as a pandas Series.
        """
        self.tree = self._build_tree(
            X, y, depth=0, attr_list=X.columns.tolist())

    def predict(self, X: pd.DataFrame) -> pd.Series:
        """
        Predict the target values for the input data.
        Args:
            X (pd.DataFrame): Input dataset as a pandas DataFrame.

        Returns:
            pd.Series: Predicted target values as a pandas Series.
        """
        predicts = pd.Series(
            [self._predict_one(row, self.tree) for _, row in X.iterrows()]
        )
        return predicts

    def _predict_one(self, x: pd.Series, root: Dict) -> int:
        """
        Utility function to predict the target value for a single row of data.
        Args:
            x (pd.Series): Input data as a pandas Series.

        Returns:
            target: Predicted target value.
        """
        if root is None:
            return None
        if root["feature"] is None:
            return root["value"]
        for attr_val, child_root, in root["children"]:
            if attr_val == x[root["feature"]]:
                return self._predict_one(x, child_root)

    def _build_tree(self, X: pd.DataFrame, y: pd.Series, depth: int = 0, parent: Dict = None, attr_list: List[str] = []) -> Dict:
        """
        Utility function to build the decision tree by reusing the recursive function.
        Args:
            X (pd.DataFrame): Input dataset as a pandas DataFrame.
            y (pd.Series): Target values as a pandas Series.
            depth (int, optional): Current depth of the node. Defaults to 0.

        Returns:
            Dict: root of the decision tree as a dictionary.
        """
        self.num_nodes += 1
        self.final_depth = max(self.final_depth, depth)
        if depth >= self.max_depth:
            return self._get_leaf(y, parent)
        if len(X) <= self.min_samples_leaf:
            return self._get_leaf(y, parent)
        if len(attr_list) == 0:
            return self._get_leaf(y, parent)
        if len(y.unique()) == 1:
            return self._get_leaf(y, parent)

        best_split_column, best_split_value, _ = self._get_best_split(
            X, y, attr_list)

        freq = np.zeros(self.num_classes)
        temp = y.to_numpy()

        for i in range(self.num_classes):
            freq[i] = np.sum(temp == i)

        dt = {
            "parent": parent,
            "feature": best_split_column,
            "value": best_split_value,
            "children": [],
            "freq": freq,
        }

        attr_list.remove(best_split_column)

        for value in X[best_split_column].unique():
            cX = X[:][X[best_split_column] == value]
            cY = y[X[best_split_column] == value]
            if len(cX) == 0:
                continue
            child_tree = self._build_tree(
                cX, cY, depth + 1, dt, attr_list.copy())
            dt["children"].append((value, child_tree))

        return dt

    def _get_best_split(self, X: pd.DataFrame, y: pd.Series, attr_list: List = None) -> Tuple[str, float, int]:
        """
        Find the best split for the current node.
        Args:
            X (pd.DataFrame): Input dataset as a pandas DataFrame.
            y (pd.Series): Target values as a pandas Series.

        Returns:
            Tuple[str, float, int]: Best split feature, value and the split score.
        """
        best_split_column = None
        best_split_value = None
        best_split_score = None

        for column in attr_list:
            if self.criterion == "gini":
                score = gini_gain(X, column, y)
            elif self.criterion == "ig":
                score = information_gain(X, column, y)
            else:
                raise ValueError("Invalid criterion")
            if best_split_score is None or score > best_split_score:
                best_split_column = column
                best_split_score = score

        best_split_value = y.value_counts().idxmax()
        return best_split_column, best_split_value, best_split_score

    def _get_leaf(self, y: pd.Series, parent: Dict = None) -> Dict:
        """
        Get the leaf node for the current node.
        Args:
            y (pd.Series): Target values as a pandas Series.

        Returns:
            Dict: Leaf node as a dictionary.
        """
        freq = np.zeros(self.num_classes)
        temp = y.to_numpy()

        freq[0] = np.sum(temp == 0, axis=0)
        freq[1] = np.sum(temp == 1, axis=0)
        freq[2] = np.sum(temp == 2, axis=0)
        freq[3] = np.sum(temp == 3, axis=0)

        return {
            "parent": parent,
            "feature": None,
            "value": y.value_counts().idxmax(),
            "children": [],
            "freq": freq,
        }

    def print_tree(self, labels: Dict[str, List[str]]) -> None:
        """
        Prints the decision tree in a graphviz format.
        Args:
            labels (Dict[str, List]): A dictionary mapping the column name to the List of column values.
        """
        gra = graphviz.Digraph(
            format="png",
            node_attr={
                "shape": "box",
                "style": "filled",
                "fillcolor": "#D3D3D3",
                "fontname": "Courier"
            }
        )
        self.__render_index = 0
        self._print_tree(self.tree, gra, labels)
        gra.render("tree.gv", view=True)

    def _print_tree(
        self,
        root: Dict,
        gra: graphviz.Digraph,
        labels: Dict[str, List[str]]
    ) -> None:
        """ Utility function to print the decision tree in a graphviz format recursively.

        Args:
            root (Dict): Root node of the decision tree.
            gra (graphviz.Digraph): Graphviz digraph object.
            labels (Dict[str, List[str]]): A dictionary mapping the column name to the List of column values.
        """
        if root["feature"] is None:
            gra.node(str(self.__render_index),
                     label=f"{labels['class'][root['value']]}")
            return

        gra.node(f"{self.__render_index}",
                 root["feature"])
        curr_index = self.__render_index
        for (attr_val, child_root) in root["children"]:
            self.__render_index += 1
            gra.edge(f"{curr_index}",
                     f"{self.__render_index}",
                     label=f"{labels[root['feature']][attr_val]}")
            self._print_tree(child_root, gra, labels)

    def prune_tree(self, method='chi', x_valid=None, y_valid=None):
        """
        Prunes the decision tree using the validation set. or using chi square test.
        Args:
            method: The method used for pruning(chi or reduced-error) Defaults to 'chi'.
            x_valid: Validation dataset if method is reduced-error.
            y_valid: Validation target values if method is reduced-error.
        """
        if method == 'chi':
            self._prune_tree2(self.tree)
        elif method == 'reduced-error':
            self._prune_tree(self.tree, x_valid, y_valid)

    def _prune_tree(self, root, x_valid, y_valid):
        """
        Utility function to prune the decision tree using reduced-error method.
        """
        if len(root["children"]) == 0:
            return
        for _, child in root["children"]:
            self._prune_tree(child, x_valid, y_valid)

        curr_acc = (self.predict(x_valid).to_numpy()
                    == y_valid.to_numpy()).mean()
        children = root["children"]
        feature = root["feature"]
        root["children"] = []
        root["feature"] = None
        self.num_nodes -= len(children)
        new_acc = (self.predict(x_valid).to_numpy()
                   == y_valid.to_numpy()).mean()
        if curr_acc > new_acc:
            root["children"] = children
            root["feature"] = feature
            self.num_nodes += len(children)
        return

    def _prune_tree2(self, root):
        """
        Utility function to prune the decision tree using chi square test.
        """
        if len(root["children"]) == 0:
            return
        for _, child in root["children"]:
            self._prune_tree2(child)

        ct_matrix = np.zeros((self.num_classes, len(root["children"])))
        for i, (_, child) in enumerate(root["children"]):
            ct_matrix[:, i] = child["freq"]

        new_ct_matrx = []

        for i in range(ct_matrix.shape[0]):
            if ct_matrix[i, :].sum() != 0:
                new_ct_matrx.append(ct_matrix[i, :])

        new_ct_matrx = np.array(new_ct_matrx)
        chi, p, dof, _ = chi2_contingency(new_ct_matrx)

        if p >= 0.05:
            self.num_nodes -= len(root["children"])
            root["children"] = []
            root["feature"] = None
        return
        ...


if __name__ == "__main__":
    dataset = CarDataset(root="./dataset")
    shuffle_dataset(dataset)
    (x_train, y_train), (x_valid, y_valid) = split_dataset(
        dataset.data, dataset.targets, 0.8)
    dt = DecisionTree(max_depth=6,
                      min_samples_leaf=1, criterion="ig")
    dt.fit(x_train, y_train)
    predicts = dt.predict(x_valid)

    print((y_valid.to_numpy() == predicts.to_numpy()).sum() / len(y_valid))
    print(dt.num_nodes)
    # dt.print_tree(dataset.metadata)

    dt.prune_tree('chi')
    predicts = dt.predict(x_valid)
    print((y_valid.to_numpy() == predicts.to_numpy()).sum() / len(y_valid))
    # dt.print_tree(dataset.metadata)
    print(dt.num_nodes)
