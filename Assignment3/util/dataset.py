import pandas as pd
import os
import urllib3
from typing import Optional, Any, Callable, Tuple, List
import torch.utils.data as data
import numpy as np
from sklearn.decomposition import PCA


class SatelliteDataset(data.Dataset):
    """
    """
    resources = [
        ('https://archive.ics.uci.edu/ml/machine-learning-databases/statlog/satimage/sat.doc', 'sat.doc'),
        ('https://archive.ics.uci.edu/ml/machine-learning-databases/statlog/satimage/sat.trn', 'sat.trn'),
        ('https://archive.ics.uci.edu/ml/machine-learning-databases/statlog/satimage/sat.tst', 'sat.tst')
    ]

    training_file = "sat.trn"
    test_file = "sat.tst"

    def __init__(
        self,
        root: str,
        train: bool = True,
        download: bool = False,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None
    ):
        super(SatelliteDataset, self).__init__()
        self.root = root
        self.transform = transform
        self.target_transform = target_transform

        if download is True:
            self.download()
        if not self._sanity_check():
            raise RuntimeError("Dataset not found, Use download = True")

        if train is True:
            data_file = self.training_file
        else:
            data_file = self.test_file
        self.data, self.targets = self._load_data(
            os.path.join(self.root, data_file))

    def __getitem__(self, idx: int) -> Tuple[Any, Any]:
        data, target = self.data[idx], int(self.targets[idx])
        data = np.array(data)

        if self.transform is not None:
            data = self.transform(data)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return data, target

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

    def _load_data(self, fname: str) -> Tuple[Any, Any]:
        df = pd.read_csv(fname, sep=' ')
        data = df.iloc[:, :-1].to_numpy().astype(np.float32)
        targets = df.iloc[:, -1].to_numpy().astype(np.int32)
        return data, targets

    def __len__(self) -> int:
        return self.data.shape[0]


class DatasetTransform:
    """
    Transformation dataset used for mean centering and scaling

    Args:
    mean: mean-array of input data
    std_dev: standard-dev of input data
    """

    def __init__(self, mean, std_dev):
        if isinstance(mean, np.ndarray) is False:
            mean = np.array(mean, dtype=np.float32)
        if isinstance(std_dev, np.ndarray):
            std_dev = np.array(std_dev, dtype=np.float32)

        self.mean = mean
        self.std_dev = std_dev + 1e-9  # for zero

    def __call__(self, x: np.ndarray) -> np.ndarray:
        return (x - self.mean) / (self.std_dev)


class MultiTransforms:
    def __init__(self, transforms: List[Callable]):
        self.transforms = transforms

    def __call__(self, x: np.ndarray) -> np.ndarray:
        for transform in self.transforms:
            x = transform(x)
        return x


class TransformPCA:
    """ 
    Wrapper for PCA to fit into transform semantics of SatelliteDataset and torch
    """

    def __init__(self, pca: PCA):
        self.pca = pca

    def __call__(self, x: np.ndarray) -> np.ndarray:
        return self.pca.transform(x).to_numpy()


if __name__ == "__main__":
    dataset = SatelliteDataset(root="../dataset",
                               train=True,
                               download=True)
    x, y = dataset[0]
    print(f"Length of dataset: {len(dataset)}")
    print(f"Sample from dataset x: {x}, y: {y}")

    mean = np.zeros_like(x)
    std_dev = np.zeros_like(x)

    for i in range(4):
        data = dataset.data[:, i::4]
        mean[i::4] = np.mean(data, dtype=np.float32)
        std_dev[i::4] = np.sqrt(np.var(data, dtype=np.float32))

    print("Mean array: {}".format(mean))
    print("Std-dev array: {}".format(std_dev))

    transform = DatasetTransform(mean, std_dev)
    test_dataset = SatelliteDataset(root="../dataset",
                                    train=False,
                                    download=False,
                                    transform=transform,
                                    target_transform=lambda x: (x-1))

    x, y = test_dataset[0]
    print(f"Transformed Sample from dataset x: {x}, y: {y}")
