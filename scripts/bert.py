from typing import List, Optional
import typer
from pathlib import Path

import torch
from torch.optim import Adam
from protein_design.learning import WarmupAnnealLR
from protein_design.splitter import random_split
from protein_design.trainer import train
from protein_design.generative import BERT
from protein_design.sequence import (
    seqs_to_integer,
    read_fasta,
    write_fasta,
)
from protein_design.splitter import random_split


app = typer.Typer()

model_params = {
    "n_head": 2,
    "d_model": 128,
    "d_k": 128,
    "d_v": 128,
    "dropout": 0.1,
    "num_mask": 9,
}


@app.command()
def train(
    fname: Path,
    save_name: Path,
    batch_size: int = 32,
    warmup_steps: int = 100,
    steps: int = 1000,
    num_heads: int = 2,
    model_dim: int = 128,
    key_dim: int = 128,
    value_dim: int = 128,
    dropout: float = 0.1,
    num_mask: int = 9,
):
    """Train a sequence attention model

    Parameters
    ----------
    fname
        Path to input fasta file containing raw or aligned sequences
    save_name
        Path to save best trained model
    batch_size, optional
        Batch size, by default 32
    warmup_steps, optional
        Number of learning rate warmup steps, following https://stackoverflow.com/q/65343377, by default 100
    steps, optional
        Number of gradient descent steps, by default 1000
    num_heads, optional
        Number of attention heads, by default 2
    model_dim, optional
        Dimension of the model representation per input token, by default 128
    key_dim, optional
        Dimension of the keys, by default 128
    value_dim, optional
        Dimension of the values, by default 128
    dropout, optional
        Dropout, by default 0.1
    num_mask, optional
        Number of consecutive tokens to mask for masked-token prediction during training, by default 9

    Example
    -------
        python3 bert.py train ../data/unaligned.fasta bert.pt
    """
    seqs = read_fasta(fname)

    X = seqs_to_integer(seqs)

    X = torch.from_numpy(X).type(torch.LongTensor)

    X_train, X_test = random_split(X)

    train_params = {
        "batch_size": batch_size,
        "lr": 0.0005,
        "weight_decay": 0.0,
        "warmup_steps": warmup_steps,
        "steps": steps,
    }
    model_params["n_head"] = num_heads
    model_params["d_model"] = model_dim
    model_params["d_k"] = key_dim
    model_params["d_v"] = value_dim
    model_params["dropout"] = dropout
    model_params["num_mask"] = num_mask

    model = BERT(**model_params)

    optimizer = Adam(
        model.parameters(),
        lr=train_params["lr"],
        weight_decay=train_params["weight_decay"],
    )

    scheduler = WarmupAnnealLR(optimizer, warmup_steps=train_params["warmup_steps"])

    train(
        model,
        X_train,
        X_test,
        save_name,
        batch_size=train_params["batch_size"],
        optimizer=optimizer,
        scheduler=scheduler,
        steps=train_params["steps"],
        pbar_increment=10,
    )


@app.command()
def sample(
    seq: str,
    saved_model: Path,
    nsample: int = 1000,
    rm_aa: str = "C,K",
    save_fname: Optional[Path] = None,
) -> Optional[List[str]]:
    """Generate sequences from a trained sequence attention model

    Parameters
    ----------
    seq
        Seed sequence where positions to generate are masked with "X"s
    saved_model
        Path to trained sequence attention model
    nsample, optional
        Number of sequences to generate, by default 1000
    rm_aa, optional
        Comma-delimited string of amino acids to ban in generated samples, by default "C,K"
    save_fname, optional
        If provided, saves a fasta file to this path. Otherwise returns a list of sampled sequences

    Returns
    -------
        When {save_fname} is not provided, returns a list of generated sequences

    Example
    -------
        python3 bert.py sample VQLQESGGGLVQAGGSLRLSCAASGSISRFNAMGWWRQAPGKEREFVARIVKGFDPVLADSVKGRFTISIDSAENTLALQMNRLKPEDTAVYYCXXXXXXXXXXXWGQGTQVTVSS bert.pt
    """
    model = BERT(**model_params)
    model.load_state_dict(torch.load(saved_model))

    sampled_seqs = model.sample(seq, n_samples=nsample, rm_aa=rm_aa)

    if save_fname is not None:
        write_fasta(save_fname, sampled_seqs)
    else:
        return sampled_seqs


if __name__ == "__main__":
    app()
