import os
from typing import Callable
import pandas as pd


class MailDataset:
    """
        Spam/Ham dataset: <https://www.kaggle.com/venky73/spam-mails-dataset/>
        Arguments:
            root [str]: Root directory of the dataset
    """
    __filename__ = "spam_ham_dataset.csv"
    __headers__ = ["_", "label", "text", "label_num"]
    __metadata__ = {
        "name": "spam_ham_dataset",
        "description": "Spam/Ham dataset",
        "length": None,
        "labels": ["ham", "spam"],
    }

    __resources__ = [
        ("https://www.kaggle.com/venky73/spam-mails-dataset/download", "archive.zip")
    ]

    @property
    def X(self) -> pd.DataFrame:
        return self.df.iloc[:, :-1]

    @X.setter
    def X(self, value: pd.DataFrame):
        self.df = value

    @property
    def Y(self) -> pd.Series:
        """
        Returns the labels of the dataset (0 for ham, 1 for spam)
        """
        return self.df.iloc[:, -1]

    def __len__(self) -> int:
        return len(self.df)

    def __init__(
        self,
        root: str,
        transform: Callable = None
    ):
        self.root = root
        self._metadata = MailDataset.__metadata__

        if not self._sanity_check():
            raise RuntimeError("Dataset file not found")

        self._read_data()
        if transform is not None:
            self.X = transform(self.X)

    def _read_data(self) -> None:
        """
        Reads the dataset from the csv file
        """
        self.df = pd.read_csv(
            os.path.join(self.root, MailDataset.__filename__))
        self.df.columns = MailDataset.__headers__
        self.df.drop(columns=["_", "label"], inplace=True)

        self._metadata["length"] = len(self.df)

    def _sanity_check(self) -> bool:
        """
        Sanity check to see if the dataset is present in the root directory
        """
        if not os.path.exists(os.path.join(self.root, MailDataset.__filename__)):
            return False
        return True


def test():
    dataset = MailDataset(root="../dataset")
    print(f"Length of dataset: {len(dataset)}")


if __name__ == "__main__":
    test()
