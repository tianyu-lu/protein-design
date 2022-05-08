import typer
from pathlib import Path
import numpy as np
import pandas as pd

from protein_design.splitter import random_split
from protein_design.sequence import seqs_to_onehot
from protein_design.discriminative import MLP
from protein_design.trainer import train


def train_mlp(
    fname: Path,
    save_name: Path,
    seqlen: int = 11,
    alphabet_size: int = 21,
    hid_dim: int = 20,
    batch_size: int = 16,
    epochs: int = 1,
):
    """train a feed-forward neural network to predict function of protein sequences

    Parameters
    ----------
    fname
        Path of input csv
    save_name
        Path to save best trained model
    seqlen, optional
        Sequence length (note all inputs must have the same length), by default 11
    alphabet_size, optional
        Number of possible amino acids, by default 21
    hid_dim, optional
        Dimension of hidden layer in neural network, by default 20
    batch_size, optional
        Batch size, by default 16
    epochs, optional
        Epochs, by default 1
    """
    df = pd.read_csv(fname, sep="\t", skiprows=1)

    df_filtered = df.loc[df["Best?"] == True]
    df_filtered = df_filtered.drop_duplicates("Slide")

    seqs = df_filtered["Slide"].to_list()
    X = seqs_to_onehot(seqs, flatten=True)

    y = df_filtered["Energy"].to_numpy()

    y = 2 * (y - np.min(y)) / (np.max(y) - np.min(y)) - 1

    X_train, y_train, X_test, y_test = random_split(X, y=y)

    model = MLP(data_dim=seqlen * alphabet_size, hid_dim=hid_dim)

    model = train(
        model,
        X_train,
        X_test,
        str(save_name),
        y_train=y_train,
        y_test=y_test,
        steps=int(len(df_filtered) / batch_size * epochs),
    )


if __name__ == "__main__":
    typer.run(train_mlp)
