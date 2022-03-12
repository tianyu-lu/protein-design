import numpy as np
import torch
from torch.utils.data import Dataset


def to_tensor(X) -> torch.tensor:
    return X if torch.is_tensor(X) else torch.from_numpy(X)


def to_numpy(X) -> np.ndarray:
    return X.detach().cpu().numpy() if torch.is_tensor(X) else np.array(X)


class ProteinData(Dataset):
    def __init__(self, X, y):
        self.X = to_tensor(X)
        self.y = to_tensor(y)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


def cycle(dataloader):
    while True:
        for data in dataloader:
            yield data
