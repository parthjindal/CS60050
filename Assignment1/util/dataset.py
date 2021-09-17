import pandas as pd
import os
import urllib3
from typing import Dict, List, Tuple


class CarDataset:
    """
    `UCI <https://archive.ics.uci.edu/ml/datasets/car+evaluation> Car Dataset`

    Arguments:
        root [str]: Root directory of the dataset
        download [bool,optional]: If true, download the dataset from the UCI website

    """
    resources = [
        ("https://archive.ics.uci.edu/ml/machine-learning-databases/car/car.data", "car.data"),
        ("https://archive.ics.uci.edu/ml/machine-learning-databases/car/car.c45-names", "car.c45-names"),
        ("https://archive.ics.uci.edu/ml/machine-learning-databases/car/car.names", "car.names"),
    ]

    headers = [
        'buying', 'maint', 'doors',
        'persons', 'lug_boot', 'safety', 'class'
    ]

    @property
    def data(self) -> pd.DataFrame:
        return self.df.iloc[:, :-1]

    @property
    def targets(self) -> pd.Series:
        return self.df.iloc[:, -1]

    @property
    def metadata(self) -> Dict:
        return self._metadata

    def __init__(
        self,
        root: str,
        download: bool = False,
    ):
        self.root: str = root
        self.filename: str = 'car.data'
        self._metadata: dict = {}

        if download:
            self.download()
        if not self._sanity_check():
            raise RuntimeError(
                'Dataset not found, Use download=True to download')
        self._read_data()
        ...

    def _read_data(self):
        """
        Utilitiy function to read the data from data file

        """
        self.df = pd.read_csv(
            os.path.join(self.root, self.filename),
            names=self.headers,
        )
        for col in self.df.columns:
            self.df[col], col_map = pd.factorize(self.df[col])
            self._metadata[col] = col_map.tolist()

    def download(self):
        """
        Download the dataset from the UCI website
        """
        if self._sanity_check():
            return

        http = urllib3.PoolManager()
        for url, fname in self.resources:
            r = http.request('GET', url)
            with open(os.path.join(self.root, fname), 'wb') as f:
                f.write(r.data)

    def _sanity_check(self):
        """
        Sanity check to see if the dataset is present in the root directory
        """
        for fname in self.resources:
            if not os.path.exists(os.path.join(self.root, fname[1])):
                return False
        return True
