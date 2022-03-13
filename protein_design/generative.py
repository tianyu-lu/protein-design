import numpy as np
import torch
import torch.nn as nn
import torch.functional as F


def KLLoss(mean, logvar):
    return torch.mean(
        -0.5 * torch.sum(1 + logvar - mean**2 - logvar.exp(), dim=1), dim=0
    )


class VAE(nn.Module):
    def __init__(self, **kwargs):
        super(VAE, self).__init__()

        self.seqlen = kwargs["seqlen"]
        self.n_tokens = kwargs["n_tokens"]
        self.latent_dim = kwargs["latent_dim"]
        self.enc_units = kwargs["enc_units"]
        self.kl_weight = kwargs["kl_weight"]

        self.encoder = nn.Sequential(
            nn.Linear(self.seqlen * self.n_tokens, self.enc_units),
            nn.ELU(),
        )
        self.mean = nn.Linear(self.enc_units, self.latent_dim)
        self.var = nn.Linear(self.enc_units, self.latent_dim)

        self.decoder = nn.Sequential(
            nn.Linear(self.latent_dim, self.enc_units),
            nn.ELU(),
            nn.Linear(self.enc_units, self.seqlen * self.n_tokens),
        )
        self.getprobs = nn.Softmax(dim=-1)

        self.recon_loss_fn = nn.MSELoss()
        self.kl_loss_fn = KLLoss

    def encode(self, x):
        z = self.encoder(x)
        mean = self.mean(z)
        logvar = self.var(z)
        return [mean, logvar]

    def decode(self, z):
        xhat = self.decoder(z).view(-1, self.seqlen, self.n_tokens)
        xhat = self.getprobs(xhat)
        return xhat

    def reparameterize(self, mean, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mean + eps * std

    def forward(self, x):
        mean, logvar = self.encode(x)
        z = self.reparameterize(mean, logvar)
        return [self.decode(z), x, mean, logvar]

    def loss(self, x, x_target):
        xhat, _, mean, logvar = self.forward(x)

        x_target = x_target.view(-1, self.seqlen, self.n_tokens)
        # x = torch.argmax(x, -1).flatten()
        # x_target = x_target.flatten(end_dim=1)
        # recon_loss = F.cross_entropy(x_target, x.type(torch.long))
        recon_loss = self.recon_loss_fn(xhat, x_target)
        kl_loss = self.kl_loss_fn(mean, logvar)

        loss = recon_loss + self.kl_weight * kl_loss

        return loss

    def sample(self, num_samples):
        z = torch.randn(num_samples, self.latent_dim).to(self.decoder.device)
        return self.decode(z)

    def reconstruct(self, x):
        recon = self.forward(x)
        return recon[0]


class Attention(nn.Module):
    def __init__(self, n_head, d_model, d_k, d_v, dropout=0.1):
        super(Attention, self).__init__()

        self.n_head = n_head
        self.d_model = d_model
        self.d_k = d_k
        self.d_v = d_v

        self.W_Q = nn.Linear(d_model, n_head * d_k)
        self.W_K = nn.Linear(d_model, n_head * d_k)
        self.W_V = nn.Linear(d_model, n_head * d_v)
        self.W_O = nn.Linear(n_head * d_v, d_model)

        self.layer_norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, q, k, v):
        batch, len_q, _ = q.size()
        batch, len_k, _ = k.size()
        batch, len_v, _ = v.size()

        Q = self.W_Q(q).view([batch, len_q, self.n_head, self.d_k])
        K = self.W_K(k).view([batch, len_k, self.n_head, self.d_k])
        V = self.W_V(v).view([batch, len_v, self.n_head, self.d_v])

        Q = Q.transpose(1, 2)
        K = K.transpose(1, 2).transpose(2, 3)
        V = V.transpose(1, 2)

        # after transpose:
        # Q: (batch, n_head, len_q, d_k)
        # K: (batch, n_head, d_k, len_q)
        # V: (batch, n_head, len_q, d_v)

        attention = torch.matmul(Q, K)
        # attention: (batch, n_head, len_q, len_q)

        attention = attention / np.sqrt(self.d_k)

        attention = F.softmax(attention, dim=-1)

        output = torch.matmul(attention, V)
        # output: (batch, n_head, len_q, d_v)

        output = output.transpose(1, 2).reshape([batch, len_q, self.d_v * self.n_head])
        # output: (batch, len_q, d_v * n_head)

        output = self.W_O(output)

        output = self.dropout(output)

        output = self.layer_norm(output + q)

        return output
