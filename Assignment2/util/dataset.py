import os
from typing import Callable, Dict
import pandas as pd


class MailDataset:
    """
        Spam/Ham dataset: <https://www.kaggle.com/venky73/spam-mails-dataset/>
        Arguments:
            root [str]: Root directory of the dataset
    """
    _filename_ = "spam_ham_dataset.csv"
    _headers_ = ["_", "label", "text", "label_num"]
    _metadata_ = {
        "name": "spam_ham_dataset",
        "description": "Spam/Ham dataset",
        "length": None,
        "labels": ["ham", "spam"],  # 0: ham, 1: spam
    }

    __resources__ = [
        ("https://www.kaggle.com/venky73/spam-mails-dataset/download", "archive.zip")
    ]

    def __len__(self) -> int:
        return len(self.X)

    @property
    def metadata(self) -> Dict:
        return self._metadata_

    def __init__(
        self,
        root: str,
        transform: Callable = None
    ):
        self.root: str = root
        self.X = None
        self.Y = None

        if not self._sanity_check():
            raise RuntimeError("Dataset file not found")

        self._read_data()
        if transform is not None:
            self.X = transform(self.X)

    def _read_data(self) -> None:
        """
        Reads the dataset from the csv file
        """
        df = pd.read_csv(
            os.path.join(self.root, MailDataset._filename_))
        df.columns = MailDataset._headers_
        df.drop(columns=["_", "label"], inplace=True)
        self._metadata_["length"] = len(df)

        self.X = df['text'].to_numpy()
        self.Y = df['label_num'].to_numpy()

    def _sanity_check(self) -> bool:
        """
        Sanity check to see if the dataset is present in the root directory
        """
        if not os.path.exists(os.path.join(self.root, MailDataset._filename_)):
            return False
        return True


def test():
    from utility import Vectorizer
    vectorizer = Vectorizer(max_features=1000)
    dataset = MailDataset(root="../dataset", transform=vectorizer)
    print(f"Length of dataset: {len(dataset)}")


if __name__ == "__main__":
    test()
