from util import create_dataloaders, seed_everything, train, test, preprocess_dataset, MultiTransforms, TransformPCA, SatelliteDataset
from models.MLP import MLP0, MLP1, MLP2
import argparse
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
from tqdm import tqdm
from torch.utils.data import DataLoader
from sklearn.decomposition import PCA
seed_everything(42)
IN_DIMS = 36
OUT_DIMS = 6

# train_loader, test_loader = create_dataloaders("./dataset", True, 1, True)
# model = MLP2(IN_DIMS, OUT_DIMS, 3, 2).to("cuda:0")
#
# losses = []
# accuracies = []
#
# for _ in range(1000):
#    loss, acc = train(model, train_loader, device='cuda:0')
#    losses.append(np.mean(loss))
#    acc = test(model, test_loader, "cuda:0",
#               fast_dev_run=True,
#               metrics=["Accuracy"])["Accuracy"]
#
#    print("Accuracy: {}".format(acc))
#    accuracies.append(acc)
#
# plt.plot(losses)
# plt.show()
#
# plt.plot(accuracies)
# plt.show()


def fit(
    model: nn.Module,
    train_loader: DataLoader,
    test_loader: DataLoader,
    args: argparse.Namespace,
):
    losses = []
    train_accuracies = []
    test_accuracies = []

    early_stop_count = 0
    early_stop_patience = 5
    early_stop_tol = 1e-3

    for i in tqdm(range(args.max_epochs)):
        loss, train_acc = train(model, train_loader,
                                lr=args.lr, device=args.device)
        loss = np.mean(loss)
        losses.append(loss)
        train_accuracies.append(train_acc)

        if args.verbose and i % args.log_freq == 0:
            print(f"Epoch: {i}/{args.max_epochs}, Loss: {loss}")
            if i % args.test_freq == 0:
                results = test(model, test_loader,
                               device=args.device)
                acc = results["Accuracy"]
                print(f"Test Accuray: {acc}")
                test_accuracies.append(acc)

        if len(losses) > 1 and abs(loss-losses[-2]) < early_stop_tol:
            early_stop_count += 1
            if (early_stop_count >= early_stop_patience):
                print(f"Early stopping after {i} epochs.")
                break
        else:
            early_stop_count = 0

    if args.plot is True:
        plt.plot(train_accuracies)
        plt.xlabel("Epochs")
        plt.ylabel("Training Accuracies")
        plt.show()

        plt.plot(test_accuracies)
        plt.xlabel("Epochs")
        plt.ylabel("Test Accuracy")
        plt.show()

    train_acc = test(model, train_loader, args.device)["Accuracy"]
    test_acc = test(model, test_loader, args.device)["Accuracy"]

    return train_acc, test_acc


def main(args: argparse.Namespace):
    models = [
        MLP0(IN_DIMS, OUT_DIMS),
        MLP1(IN_DIMS, OUT_DIMS, 2),
        MLP1(IN_DIMS, OUT_DIMS, 6),
        MLP2(IN_DIMS, OUT_DIMS, 2, 3),
        MLP2(IN_DIMS, OUT_DIMS, 3, 2)]
    lrs = []
    model = MLP2(IN_DIMS, OUT_DIMS, 3, 2).to("cuda:0")

    train_dataset = SatelliteDataset("./dataset")
    mtransform, mtarget_transform = preprocess_dataset(train_dataset)

    pca = PCA(2)
    mpca.fit(train_dataset.data)

    pca_transform = TransformPCA(pca)
    data_transforms = MultiTransforms([mtransform, pca_transform])

    train_dataset.transform = data_transforms
    train_dataset.target_transform = mtarget_transform

    test_dataset = SatelliteDataset(
        "./dataset", False, transform=data_transforms, target_transform=mtarget_transform)

    train_loader, test_loader = create_dataloaders(
        train_dataset, test_dataset, shuffle=True, batch_size=1)

    args = argparse.Namespace(max_epochs=200, lr=1e-3, device="cuda:0",
                              verbose=True, log_freq=1, test_freq=1,
                              plot=True)
    train_acc, test_acc = fit(model, train_loader, test_loader, args)
    print(f"Final training accuracy: {train_acc}, test accuracy: {test_acc}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cuda", action="store_true")
    parser.add_argument("--verbose", action="store_true")

    args = parser.parse_args()

    main(args)
