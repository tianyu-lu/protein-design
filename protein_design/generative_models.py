import torch
import torch.nn as nn
import torch.nn.functional as F


class VAE(nn.Module):
    def __init__(self, **kwargs):
        super(VAE, self).__init__()

        self.seqlen = kwargs["seqlen"]
        self.n_tokens = kwargs["n_tokens"]
        self.latent_dim = kwargs["latent_dim"]
        self.enc_units = kwargs["enc_units"]

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

    def forward(self, x, **kwargs):
        mean, logvar = self.encode(x)
        z = self.reparameterize(mean, logvar)
        return [self.decode(z), x, mean, logvar]

    def loss(self, *args, **kwargs):
        xhat = args[0]
        x = args[1]
        mean = args[2]
        logvar = args[3]

        kl_weight = kwargs["kl_weight"]

        x = x.view(-1, self.seqlen, self.n_tokens)
        # x = torch.argmax(x, -1).flatten()
        # xhat = xhat.flatten(end_dim=1)
        # recon_loss = F.cross_entropy(xhat, x.type(torch.long))
        recon_loss = F.mse_loss(x, xhat)

        kl_loss = torch.mean(
            -0.5 * torch.sum(1 + logvar - mean**2 - logvar.exp(), dim=1), dim=0
        )

        loss = recon_loss + kl_weight * kl_loss

        return {"loss": loss, "recon_loss": recon_loss, "kl_loss": -kl_loss}

    def sample(self, num_samples, device, **kwargs):
        z = torch.randn(num_samples, self.latent_dim).to(device)
        return self.decode(z)

    def reconstruct(self, x, **kwargs):
        recon = self.forward(x)
        return recon[0]