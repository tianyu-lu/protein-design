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

fp = "../data/aligned.fasta"
seqs = read_fasta(fp)

X = seqs_to_onehot(seqs, flatten=False)
X = trim_gaps(X)

B, L, D = X.shape
X = X.reshape(B, L * D)
X_train, X_test = random_split(X)


train_params = {
    "batch_size": 16,
    "lr": 0.005,
    "weight_decay": 0.0,
    "scheduler_gamma": 0.95,
    "steps": 2000,
}
model_params = {
    "seqlen": L,
    "n_tokens": 21,
    "latent_dim": 20,
    "enc_units": 50,
    "kl_weight": 64 / len(X_train),
}

model = VAE(**model_params)

optimizer = Adam(
    model.parameters(), lr=train_params["lr"], weight_decay=train_params["weight_decay"]
)
scheduler = ExponentialLR(optimizer, gamma=train_params["scheduler_gamma"])

train(
    model,
    X_train,
    X_test,
    "vae.pt",
    batch_size=train_params["batch_size"],
    optimizer=optimizer,
    scheduler=scheduler,
    steps=train_params["steps"],
)

seq_probs = model.sample(1000)

sampled_seqs = probs_to_seqs(seq_probs, sample=False)
write_fasta("vae.fasta", sampled_seqs)
