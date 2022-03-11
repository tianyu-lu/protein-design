import torch.nn as nn


class MLP(nn.Module):
    def __init__(self, **kwargs):
        super(MLP, self).__init__()

        self.data_dim = kwargs["data_dim"]
        self.hid_dim = kwargs["hid_dim"]

        self.model = nn.Sequential(
            nn.Linear(self.data_dim, self.hid_dim),
            nn.ELU(),
            nn.Linear(self.hid_dim, 1),
            nn.Tanh(),
        )

    def forward(self, x):
        return self.model(x)


import math
import torch
import gpytorch
from matplotlib import pyplot as plt
from protein_design.constants import BLOSUM

from pathlib import Path
import pandas as pd
from protein_design.sequence import seqs_to_onehot
from protein_design.splitter import random_split


fname = Path("/home/tianyulu/Downloads/absolut/4K3J_A/4K3J_AFinalBindings_Process_1_Of_32.txt")

df = pd.read_csv(fname, sep='\t', skiprows=1)

df_filtered = df.loc[df["Best?"] == True]
df_filtered = df_filtered.drop_duplicates("Slide")
df_filtered = df_filtered.iloc[list(range(200))]

seqs = df_filtered["Slide"].to_list()
X = seqs_to_onehot(seqs, flatten=False)

y = energies = df_filtered["Energy"].to_numpy()


training_iter = 50

import numpy as np

X = np.argmax(X, axis=-1)

X_train, y_train, X_test, y_test = random_split(X, y=y)

X_train = torch.from_numpy(X_train)
y_train = torch.from_numpy(y_train)

# Wrap training, prediction and plotting from the ExactGP-Tutorial into a function,
# so that we do not have to repeat the code later on
def train(model, likelihood, training_iter=training_iter):
    # Use the adam optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=0.1)  # Includes GaussianLikelihood parameters

    # "Loss" for GPs - the marginal log likelihood
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)

    for i in range(training_iter):
        # Zero gradients from previous iteration
        optimizer.zero_grad()
        # Output from model
        output = model(X_train)
        # Calc loss and backprop gradients
        loss = -mll(output, y_train)
        loss.backward()
        optimizer.step()
        print(loss.item())


def predict(model, likelihood, test_x = torch.linspace(0, 1, 51)):
    model.eval()
    likelihood.eval()
    # Make predictions by feeding model through likelihood
    with torch.no_grad(), gpytorch.settings.fast_pred_var():
        # Test points are regularly spaced along [0,1]
        return likelihood(model(test_x))

class BlosumKernel(gpytorch.kernels.Kernel):
    has_lengthscale = True

    # this is the kernel function
    def forward(self, x1, x2, **params):
        c = torch.tensor(1.0)
        beta = torch.tensor(0.1)
        kappa = torch.zeros(len(x1), len(x2))
        for i, xi in enumerate(x1):
            for j, xj in enumerate(x2):
                kij = torch.prod(BLOSUM[(xi, xj)] ** beta)
                kii = torch.prod(BLOSUM[(xi, xi)] ** beta)
                kjj = torch.prod(BLOSUM[(xj, xj)] ** beta)

                k = kij / (torch.sqrt(kii * kjj))

                k = torch.exp(c * k)
                k = (k + c) / self.lengthscale

                kappa[i, j] = k

        return kappa

# Use the simplest form of GP model, exact inference
class BlosumGPModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood):
        super().__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = BlosumKernel()

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

likelihood = gpytorch.likelihoods.GaussianLikelihood()
# initialize the new model
model = BlosumGPModel(X_train, y_train, likelihood)
model.double()

# set to training mode and train
model.train()
likelihood.train()
train(model, likelihood)

# Get into evaluation (predictive posterior) mode and predict
model.eval()
likelihood.eval()
observed_pred = predict(model, likelihood)

print("done")
