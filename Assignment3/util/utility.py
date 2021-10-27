import os
import random
import sys
from typing import List, Tuple, Optional
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torch.nn.functional as F
from .dataset import SatelliteDataset as Dataset, DatasetTransform
from .metrics import get_metrics


NUM_SPECTRAL_BANDS = 4


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)


def create_dataloaders(
    root_dir: str,
    shuffle: bool = True,
    batch_size: int = 1,
    test: bool = True,
) -> Tuple[DataLoader, Optional[DataLoader]]:

    train_dataset = Dataset(root_dir, True)

    mean = np.empty((train_dataset.data.shape[1],), dtype=np.float32)
    std_dev = np.empty_like(mean)

    for i in range(NUM_SPECTRAL_BANDS):
        data = train_dataset.data[:, i::NUM_SPECTRAL_BANDS]
        mean[i::NUM_SPECTRAL_BANDS] = np.mean(data)
        std_dev = np.sqrt(np.var(data))

    transform = DatasetTransform(mean, std_dev)

    def target_transform(x):
        return x-1 if x < 6 else x-2

    train_dataset.transform = transform
    train_dataset.target_transform = target_transform

    train_loader = DataLoader(train_dataset, batch_size=batch_size,
                              shuffle=shuffle)
    if test is True:
        test_dataset = Dataset(root_dir, False,
                               transform=transform,
                               target_transform=target_transform)
        test_loader = DataLoader(test_dataset, batch_size=batch_size)
        return train_loader, test_loader
    else:
        return train_loader


@torch.enable_grad()
def train(
    model: nn.Module,
    dataloader: DataLoader,
    lr: float = 1e-3,
    device: str = "cpu",
    fast_dev_run=False,
):
    model.train(True)
    optimizer = optim.Adam(params=model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()
    losses = []

    num_data = 0
    num_correct = 0

    for (x, y) in dataloader:
        x = x.to(device)
        y = y.to(device)
        y_pred = model(x)

        loss = loss_fn(y_pred, y)
        loss.backward()
        optimizer.step()

        losses.append(loss.item())

        num_correct += torch.sum(torch.argmax(y_pred, 1) == y).float()
        num_data += len(x)

        if fast_dev_run is True:
            break
    return losses, num_correct.item() / num_data


def test(
    model: nn.Module,
    dataloader: DataLoader,
    device: str = "cpu",
    fast_dev_run: bool = False,
    metrics: List[str] = ["Accuracy"]
):
    predicts = []
    y_true = []
    for (x, y) in dataloader:
        x = x.to(device)
        y = y.to(device)
        out = model(x)
        y_pred = torch.argmax(out, 1)
        predicts.extend(y_pred.tolist())
        y_true.extend(y.tolist())

        if fast_dev_run is True:
            break

    results = get_metrics(predicts, y_true, metrics)
    return results
