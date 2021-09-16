import pandas as pd
from util import information_gain, gini_gain
import graphviz
from typing import List, Tuple, Dict
from util import CarDataset


class DecisionTree():
    """
    Decision Tree model for supervised learning
    Args:
        max_depth (int): Maximum depth of any branch in the Decision tree.Defaults to 10.
        min_samples_leaf (int): Minimum no. of samples for further splitting nodes else treated as a leave. Defaults to 1.
        criterion (str): Criteria to select the attribute at a node(['gini', 'entropy']). Defaults to "gini".
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

    def _build_tree(self, X: pd.DataFrame, y: pd.Series, depth: int = 0, attr_list: List[str] = []) -> Dict:
        """
        Utility function to build the decision tree by reusing the recursive function.
        Args:
            X (pd.DataFrame): Input dataset as a pandas DataFrame.
            y (pd.Series): Target values as a pandas Series.
            depth (int, optional): Current depth of the node. Defaults to 0.

        Returns:
            Dict: root of the decision tree as a dictionary.
        """
        self.final_depth = max(self.final_depth, depth)
        if depth >= self.max_depth:
            return self._get_leaf(y)
        if len(X) <= self.min_samples_leaf:
            return self._get_leaf(y)
        if len(attr_list) == 0:
            return self._get_leaf(y)
        if len(y.unique()) == 1:
            return self._get_leaf(y)

        best_split_column, best_split_value, _ = self._get_best_split(
            X, y, attr_list)

        dt = {
            "feature": best_split_column,
            "value": best_split_value,
            "children": []
        }

        attr_list.remove(best_split_column)

        for value in X[best_split_column].unique():
            cX = X[:][X[best_split_column] == value]
            cY = y[X[best_split_column] == value]
            if len(cX) == 0:
                continue
            child_tree = self._build_tree(cX, cY, depth + 1, attr_list.copy())
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
            elif self.criterion == "entropy":
                score = information_gain(X, column, y)
            else:
                raise ValueError("Invalid criterion")
            if best_split_score is None or score > best_split_score:
                best_split_column = column
                best_split_score = score

        best_split_value = y.value_counts().idxmax()
        return best_split_column, best_split_value, best_split_score

    def _get_leaf(self, y: pd.Series) -> Dict:
        """
        Get the leaf node for the current node.
        Args:
            y (pd.Series): Target values as a pandas Series.

        Returns:
            Dict: Leaf node as a dictionary.
        """
        return {
            "feature": None,
            "value": y.value_counts().idxmax(),
            "children": []
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


if __name__ == "__main__":
    dataset = CarDataset(root="./dataset")
    X, y = dataset.data, dataset.targets
    dt = DecisionTree(max_depth=5,
                      min_samples_leaf=10, criterion="gini")
    dt.fit(X, y)
    predicts = dt.predict(X)
    print((y == predicts).sum() / len(y))
    dt.print_tree(dataset.metadata)
