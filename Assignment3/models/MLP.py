import torch
import torch.nn as nn
import torch.nn.functional as F


class MLP0(nn.Module):
    def __init__(self, in_dims, out_dims,):
        super(MLP0, self).__init__()
        self.layer = nn.Linear(in_dims, out_dims)

    def forward(self, x):
        return self.layer(x)


class MLP1(nn.Module):
    def __init__(self, in_dims, out_dims, nodes):
        super(MLP1, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(in_dims, nodes),
            nn.ReLU(),
            nn.Linear(nodes, out_dims),
        )

    def forward(self, x):
        return self.layers(x)


class MLP2(nn.Module):
    def __init__(self, in_dims, out_dims, n1, n2):
        super(MLP2, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(in_dims, n1),
            nn.ReLU(),
            nn.Linear(n1, n2),
            nn.ReLU(),
            nn.Linear(n2, out_dims),
        )

    def forward(self, x):
        return self.layers(x)


if __name__ == "__main__":
    model1 = MLP0(100, 10)
    x = torch.randn((100, 100))
    y = model1(x)
    print("Test 0 passed!")

    model2 = MLP1(100, 10, 20)
    y = model2(x)
    print("Test 1 passed!")

    model3 = MLP2(100, 10, 20, 15)
    y = model3(x)
    print("Test 2 passed!")
