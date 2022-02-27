import torch.nn as nn


class MLP(nn.Module):
    def __init__(self, **kwargs):
        super(MLP, self).__init__()

        self.data_dim = kwargs["data_dim"]
        self.hid_dim = kwargs["hid_dim"]
        self.num_layers = kwargs["num_layers"]

        self.model = nn.Sequential(
            nn.Linear(self.data_dim, self.hid_dim),
            nn.ELU(),
            nn.Linear(self.hid_dim, 1),
            nn.Tanh(),
        )

    def forward(self, x):
        return self.model(x)
