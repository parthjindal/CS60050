import pandas as pd
from util import information_gain, gini_gain
import graphviz
from typing import List, Tuple, Dict
from util import CarDataset


class DecisionTree:
    def __init__(
        self,
        max_depth: int = 10,
        min_samples_leaf: int = 1,
        criterion: str = "gini"
    ) -> None:
        """[summary]

        Args:
            dataset (Any): [description]
            max_depth (int, optional): [description]. Defaults to 10.
            min_samples_leaf (int, optional): [description]. Defaults to 1.
            criterion (str, optional): [description]. Defaults to "gini".
        """
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.criterion = criterion
        self.tree = None
        self.final_depth = 0

    def fit(self, X: pd.DataFrame, y: pd.Series) -> None:
        """[summary]

        Args:
            X (pd.DataFrame): [description]
            y (pd.Series): [description]
        """
        self.tree = self._build_tree(
            X, y, depth=0, attr_list=X.columns.tolist())

    def predict(self, X: pd.DataFrame) -> pd.Series:
        """[summary]

        Args:
            X (pd.DataFrame): [description]

        Returns:
            pd.Series: [description]
        """
        predicts = pd.Series(
            [self._predict_one(row, self.tree) for _, row in X.iterrows()]
        )
        return predicts

    def _predict_one(self, x: pd.Series, root: Dict) -> int:
        """[summary]

        Args:
            x (pd.Series): [description]

        Returns:
            str: [description]
        """
        if root is None:
            return None
        if root["feature"] is None:
            return root["value"]
        for attr_val, child_root, in root["children"]:
            if attr_val == x[root["feature"]]:
                return self._predict_one(x, child_root)

    def _build_tree(self, X: pd.DataFrame, y: pd.Series, depth: int = 0, attr_list: List = None) -> Dict:
        """[summary]

        Args:
            X (pd.DataFrame): [description]
            y (pd.Series): [description]
            depth (int, optional): [description]. Defaults to 0.

        Returns:
            Any: [description]
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

        best_split_column, best_split_value, best_score = self._get_best_split(
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
        """[summary]

        Args:
            X (pd.DataFrame): [description]
            y (pd.Series): [description]

        Returns:
            Tuple[str, float, int]: [description]
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
                best_split_value = X[column].value_counts().idxmax()

        return best_split_column, best_split_value, best_split_score

    def _get_leaf(self, y: pd.Series) -> Dict:
        """[summary]

        Args:
            y (pd.Series): [description]

        Returns:
            Dict: [description]
        """
        return {
            "feature": None,
            "value": y.value_counts().idxmax(),
            "children": []
        }

    def print_tree(self, root: Dict, labels) -> None:
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
        self._print_tree(root, gra, labels)
        gra.render("tree.gv", view=True)

    def _print_tree(self, root: Dict, gra: graphviz.Digraph, labels) -> None:
        if root["feature"] is None:
            gra.node(
                str(self.__render_index),
                label=f"{labels['class'][root['value']]}",
            )
            return
        gra.node(
            f"{self.__render_index}",
            root["feature"]
        )
        parent_index = self.__render_index

        for _, (attr_val, child_root) in enumerate(root["children"]):
            self.__render_index += 1
            gra.edge(
                f"{parent_index}",
                f"{self.__render_index}",
                label=f"{labels[root['feature']][attr_val]}")
            self._print_tree(child_root, gra, labels)


if __name__ == "__main__":
    dataset = CarDataset(root="./dataset")
    X, y = dataset.data, dataset.targets
    dt = DecisionTree(max_depth=3,
                      min_samples_leaf=10, criterion="gini")
    dt.fit(X, y)
    predicts = dt.predict(X)
    print((y == predicts).sum() / len(y))
    dt.print_tree(dt.tree, dataset.metadata)
