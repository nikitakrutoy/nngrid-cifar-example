from torchvision.models import resnet18
from torchvision.datasets import CIFAR10
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from google.cloud import storage
import numpy as np
import pickle
import os
import torch
import io

class Model(torch.nn.Module):
    def __init__(self, pretrained=False):
        super().__init__()
        self.resnet = resnet18(pretrained=pretrained)
        self.classifier = torch.nn.Sequential(
            torch.nn.LeakyReLU(),
            torch.nn.BatchNorm1d(num_features=1000),
            torch.nn.Linear(1000, 10),
        )
    
    def forward(self, X):
        return self.classifier(self.resnet(X))


class CIFAR10Dataset(CIFAR10):
    def __init__(self, root_path):
        folder = os.path.join(root_path, "data/")
        super().__init__(
            root=folder,
            transform=transforms.Compose([
                transforms.Resize((256, 256)),
                transforms.ToTensor()
            ])
        )
    def __getitem__(self, index):
        if type(index) == slice:
            start = 0 if index.start is None else index.start
            stop = 0 if index.stop is None else index.stop
            X, y = list(zip(*[self[i] for i in range(start, stop)]))
            return torch.stack(X), torch.Tensor(y).view(-1, 1)

        return super().__getitem__(index)

class Loss(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, logits, y):
        y_pred = self.model(logits)
        return torch.nn.CrossEntropyLoss()(y_pred, y.view(-1).long())


Opt = torch.optim.Adam
DatasetClass = CIFAR10Dataset




