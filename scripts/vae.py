from typing import List, Optional
import typer
from pathlib import Path

import torch
from torch.optim import Adam
from torch.optim.lr_scheduler import ExponentialLR

from protein_design.splitter import random_split
from protein_design.trainer import train
from protein_design.generative import VAE
from protein_design.sequence import (
    trim_gaps,
    probs_to_seqs,
    seqs_to_onehot,
    read_fasta,
    write_fasta,
)
from protein_design.splitter import random_split

app = typer.Typer()

model_params = {
    "seqlen": 100,
    "n_tokens": 21,
    "latent_dim": 20,
    "hidden_dim": 50,
    "kl_weight": 0.1,
}


@app.command()
def train(
    fname: Path,
    save_name: Path,
    batch_size: int = 16,
    lr: float = 0.005,
    steps: int = 1000,
    latent_dim: int = 20,
    hidden_dim: int = 50,
):
    """Train a variational autoencoder on protein sequences

    Parameters
    ----------
    fname
        Path to input fasta file containing aligned sequences
    save_name
        Path to save best trained model
    batch_size, optional
        Batch size, by default 16
    lr, optional
        Learning rate, by default 0.005
    steps, optional
        Number of gradient descent steps
    latent_dim, optional
        Dimension of VAE latent space, by default 20
    hidden_dim, optional
        Dimension of hidden layer in encoder and decoder, by default 50

    Example
    -------
        python3 vae.py train ../data/aligned.fasta vae.pt --latent-dim 2
    """
    seqs = read_fasta(fname)

    X = seqs_to_onehot(seqs, flatten=False)
    X = trim_gaps(X)

    B, L, D = X.shape
    X = X.reshape(B, L * D)
    X_train, X_test = random_split(X)

    train_params = {
        "batch_size": batch_size,
        "lr": lr,
        "weight_decay": 0.0,
        "scheduler_gamma": 0.95,
        "steps": steps,
    }

    model_params["seqlen"] = L
    model_params["latent_dim"] = latent_dim
    model_params["hidden_dim"] = hidden_dim
    model_params["kl_weight"] = 64 / len(X_train)

    model = VAE(**model_params)

    optimizer = Adam(
        model.parameters(),
        lr=train_params["lr"],
        weight_decay=train_params["weight_decay"],
    )
    scheduler = ExponentialLR(optimizer, gamma=train_params["scheduler_gamma"])

    train(
        model,
        X_train,
        X_test,
        str(save_name),
        batch_size=train_params["batch_size"],
        optimizer=optimizer,
        scheduler=scheduler,
        steps=train_params["steps"],
    )


@app.command()
def sample(
    saved_model: Path,
    nsample: int = 1000,
    save_fname: Optional[Path] = None,
) -> Optional[List[str]]:
    """_summary_

    Parameters
    ----------
    saved_model
        Path to trained VAE
    nsample, optional
        Number of sequences to generate from the trained VAE, by default 1000
    save_fname, optional
        If provided, saves a fasta file to this path. Otherwise returns a list of sampled sequences

    Returns
    -------
        When {save_fname} is not provided, returns a list of generated sequences

    Example
    -------
        python3 vae.py sample vae.pt --nsample 500 --save-fname vae.fasta
    """
    model = model = VAE(**model_params)
    model.load_state_dict(torch.load(saved_model))

    seq_probs = model.sample(nsample)

    sampled_seqs = probs_to_seqs(seq_probs, sample=False)

    if save_fname is not None:
        write_fasta(save_fname.with_suffix(".fasta"), sampled_seqs)
    else:
        return sampled_seqs


if __name__ == "__main__":
    app()
