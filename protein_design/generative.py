import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from protein_design.data import to_tensor
from protein_design.constants import AA, AA_IDX, device
from protein_design.sequence import integer_to_seqs, seqs_to_integer


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
        z = (
            torch.randn(num_samples, self.latent_dim)
            .type(torch.DoubleTensor)
            .to(device)
        )
        return self.decode(z).cpu().detach().numpy()

    def reconstruct(self, x):
        recon = self.forward(x)
        return recon[0]


class Attention(nn.Module):
    def __init__(self, **kwargs):
        super(Attention, self).__init__()

        self.n_head = kwargs["n_head"]
        self.d_model = kwargs["d_model"]
        self.d_k = kwargs["d_k"]
        self.d_v = kwargs["d_v"]

        self.W_Q = nn.Linear(self.d_model, self.n_head * self.d_k)
        self.W_K = nn.Linear(self.d_model, self.n_head * self.d_k)
        self.W_V = nn.Linear(self.d_model, self.n_head * self.d_v)
        self.W_O = nn.Linear(self.n_head * self.d_v, self.d_model)

        self.layer_norm = nn.LayerNorm(self.d_model)
        self.dropout = nn.Dropout(kwargs["dropout"])

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


def label_smoothed_nll_loss(lprobs, target, epsilon: float = 1e-8, ignore_index=None):
    """Adapted from fairseq

    Parameters
    ----------
    lprobs
        Log probabilities of amino acids per position
    target
        Target amino acids encoded as integer indices
    epsilon
        Smoothing factor between 0 and 1, by default 1e-8
    ignore_index, optional
        Amino acid (encoded as integer) to ignore, by default None

    Returns
    -------
        Negative log-likelihood loss
    """
    nll_loss = -lprobs.gather(dim=-1, index=target)
    smooth_loss = -lprobs.sum(dim=-1, keepdim=True)
    if ignore_index is not None:
        pad_mask = target.eq(ignore_index)
        nll_loss.masked_fill_(pad_mask, 0.0)
        smooth_loss.masked_fill_(pad_mask, 0.0)
    else:
        nll_loss = nll_loss.squeeze(-1)
        smooth_loss = smooth_loss.squeeze(-1)
    nll_loss = nll_loss.sum()
    smooth_loss = smooth_loss.sum()
    eps_i = epsilon / lprobs.size(-1)
    loss = (1.0 - epsilon) * nll_loss + eps_i * smooth_loss
    return loss


def positional_embedding(d_model: int, length: int) -> torch.tensor:
    result = -torch.exp(torch.arange(0, d_model, 2) * (np.log(10000) / d_model))

    numbers = torch.arange(0, length)
    numbers = numbers.unsqueeze(0)
    numbers = numbers.unsqueeze(2)

    result = numbers * result
    result = torch.cat((torch.sin(result), torch.cos(result)), 2)
    result = to_tensor(result)

    return nn.Parameter(result, requires_grad=True)


class BERT(nn.Module):
    def __init__(self, **kwargs):
        super(BERT, self).__init__()

        self.d_model = kwargs["d_model"]

        self.embed = nn.Embedding(len(AA) + 1, self.d_model)
        self.positional_embedding = None

        self.layer1 = Attention(**kwargs)
        self.layer2 = Attention(**kwargs)
        self.linear = nn.Linear(self.d_model, len(AA))

        self.num_mask = kwargs["num_mask"]

    def forward(self, x):
        embeds = self.embed(x)
        if self.positional_embedding is None:
            self.positional_embedding = positional_embedding(self.d_model, x.shape[1])
        embeds += self.positional_embedding

        h1 = self.layer1(embeds, embeds, embeds)
        h2 = self.layer2(h1, h1, h1)

        B, L, D = h2.shape
        logits = self.linear(h2.view(B * L, D)).view(B, L, len(AA))
        lprobs = F.log_softmax(logits, dim=-1)

        return lprobs

    def loss(self, x, _):
        B, L = x.shape
        start_idx = np.random.choice(range(L - self.num_mask))
        masked_idx = tuple(range(start_idx, start_idx + self.num_mask))

        x_target = x[:, masked_idx]
        x[:, masked_idx] = 21

        lprobs = self.forward(x)[:, masked_idx]
        loss = label_smoothed_nll_loss(
            lprobs.view(-1, lprobs.size(-1)), x_target.view(-1, 1), ignore_index=20
        )
        return loss / (B * self.num_mask)

    def sample(self, seq, n=1000, rm_aa=""):
        masked = [i for i in range(len(seq)) if seq[i] == 'X']
        x = seqs_to_integer([seq])
        x = [aa if i not in masked else 21 for i, aa in enumerate(x[0])]
        x = np.expand_dims(np.array(x), 0)
        x = torch.from_numpy(x).type(torch.LongTensor)

        with torch.no_grad():
            lprobs = self.forward(x).cpu().detach().numpy()

        def _sample(i) -> int:
            probs = np.exp(lprobs[i])
            probs[20] = 0
            probs[21] = 0
            for aa in rm_aa.split(","):
                probs[AA_IDX[aa.upper()]] = 0
            return np.random.choice(22, p=probs)

        x = seqs_to_integer([seq])
        all_sampled = []
        for _ in range(n):
            sampled = []
            for i in range(len(seq)):
                if i in masked:
                    sampled.append(_sample(i))
                else:
                    sampled.append(x[i])
            all_sampled.append(sampled)
        
        return integer_to_seqs(np.array(all_sampled))
