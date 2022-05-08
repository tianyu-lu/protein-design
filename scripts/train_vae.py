import typer
from pathlib import Path

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


def train_vae(
    fname: Path,
    save_name: Path,
    batch_size: int = 16,
    lr: float = 0.005,
    steps: int = 1000,
    latent_dim: int = 20,
    hidden_dim: int = 50,
    nsample: int = 1000,
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
    nsample, optional
        Number of sequences to generate from the trained VAE, by default 1000
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
    model_params = {
        "seqlen": L,
        "n_tokens": 21,
        "latent_dim": latent_dim,
        "hidden_dim": hidden_dim,
        "kl_weight": 64 / len(X_train),
    }

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

    seq_probs = model.sample(nsample)

    sampled_seqs = probs_to_seqs(seq_probs, sample=False)
    write_fasta(save_name.with_suffix(".fasta"), sampled_seqs)


if __name__ == "__main__":
    typer.run(train_vae)
