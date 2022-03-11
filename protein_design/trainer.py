from typing import Callable

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from tqdm import tqdm
import pandas as pd
import numpy as np
import plotly.express as px

from protein_design.data import SeqfuncData, SequenceData, cycle
from protein_design.evaluator import regression_metrics


def forward_pass(model: nn.Module, X: torch.tensor, y: torch.tensor) -> torch.tensor:
    """One forward pass of {model} with input {X} and desired output {y}

    Parameters
    ----------
    model
        PyTorch model
    X
        Inputs to the model
    y
        Desired output of the model

    Returns
    -------
        Loss
    """
    yhat = model(X).squeeze()
    loss_fn = nn.MSELoss()
    loss = loss_fn(yhat, y)
    return loss


def training_step(model: nn.Module, data_generator: Callable, optimizer, scheduler):
    X, y = next(data_generator)

    loss = forward_pass(model, X, y)

    loss.backward()
    optimizer.step()
    scheduler.step()

    return loss.item()


def validation_step(model: nn.Module, data_generator: Callable):
    X, y = next(data_generator)

    with torch.no_grad():
        loss = forward_pass(model, X, y)

    return loss.item()


def train(
    model,
    X_train,
    X_test,
    save_fname,
    y_train=None,
    y_test=None,
    steps=100,
    batch_size=32,
    optimizer=None,
    scheduler=None,
):
    if optimizer is None:
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    if scheduler is None:
        scheduler = torch.optim.lr_scheduler.CyclicLR(
            optimizer, 1e-5, 1e-4, cycle_momentum=False
        )

    if y_train is None:
        train_dataset = SequenceData(X_train)
        test_dataset = SequenceData(X_test)
    else:
        train_dataset = SeqfuncData(X_train, y_train)
        test_dataset = SeqfuncData(X_test, y_test)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

    train_cycle = cycle(train_loader)
    test_cycle = cycle(test_loader)

    train_losses, val_losses = [], []

    best_model = None

    best_val_loss = 1e9
    progress_format = "train loss: {:.6f} val loss: {:.6f}"
    with tqdm(total=steps, desc=progress_format.format(0, 0)) as pbar:
        for i in range(steps):
            train_loss = training_step(model, train_cycle, optimizer, scheduler)
            val_loss = validation_step(model, test_cycle)
            train_losses.append(train_loss)
            val_losses.append(val_loss)
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(model.state_dict(), save_fname)
                best_model = model
            if (i + 1) % 1000 == 0:
                pbar.set_description(progress_format.format(train_loss, val_loss))
                pbar.update(1000)

    df_result = pd.DataFrame()
    df_result["steps"] = list(range(steps)) + list(range(steps))
    df_result["loss"] = train_losses + val_losses
    df_result["stage"] = ["train"] * steps + ["val"] * steps
    fig = px.line(df_result, x="steps", y="loss", color="stage")
    fig.show()

    if y_train is not None:
        y_true, y_pred = [], []
        for x, y in test_loader:
            y = y.detach().cpu().numpy()
            y_true.extend(list(y))
            pred = model(x).detach().cpu().numpy().squeeze()
            y_pred.extend(list(pred))
        regression_metrics(np.array(y_true), np.array(y_pred), plot=True)

    return best_model
