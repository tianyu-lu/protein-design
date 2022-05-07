from typing import Union
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

from protein_design.constants import device


def to_tensor(X: Union[np.ndarray, torch.tensor]) -> torch.tensor:
    """Convert to torch tensor

    Parameters
    ----------
    X
        Numpy array or torch tensor

    Returns
    -------
        X as torch tensor
    """
    return X.to(device) if torch.is_tensor(X) else torch.from_numpy(X).to(device)


def to_numpy(X: Union[np.ndarray, torch.tensor]) -> np.ndarray:
    """Convert to numpy array

    Parameters
    ----------
    X
        Numpy array or torch tensor

    Returns
    -------
        X as numpy array
    """
    return X.detach().cpu().numpy() if torch.is_tensor(X) else np.array(X)


class ProteinData(Dataset):
    def __init__(self, X, y):
        self.X = to_tensor(X)
        self.y = to_tensor(y)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


def cycle(dataloader: DataLoader):
    """A generator wrapper around a PyTorch DataLoader

    Parameters
    ----------
    dataloader
        PyTorch DataLoader object

    Yields
    ------
        One batch of data to be used for training/validation/testing
    """
    while True:
        for data in dataloader:
            yield data
