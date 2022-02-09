import logging
from math import pi
from types import SimpleNamespace

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.optim import Adam

from model import MLPEnsemble


def main():
    x_low, x_high = -2 * pi, 2 * pi
    data_size = 1000

    x = np.linspace(x_low, x_high, data_size)
    y = np.sin(x)

    noise1 = np.random.randn(*x.shape) * np.sqrt(np.abs(np.sin(1.5 * x + pi / 8)))
    noise2 = np.random.randn(*x.shape)

    noise1_strength = 0.15
    noise2_strength = 0.2

    print("")
    plt.figure(figsize=(16, 8))
    plt.plot(x, y)
    plt.plot(x, y + noise1_strength * noise1, ".")
    plt.plot(x, y + noise2_strength * noise2, "o")
    # plt.plot(x, y, x, y + noise1_strength * noise1, ".", x, y + noise2_strength * noise2, "o")
    plt.show()
    print("")

    pass


def gauss_nll_ensemble_loss(ensemble_out, target):
    losses = []
    for mean_model, log_std_model in ensemble_out:
        var = torch.exp(2 * log_std_model)
        model_loss = F.gaussian_nll_loss(mean_model, target, var)
        losses.append(model_loss)

    total_loss = torch.stack(losses).mean()
    return total_loss


def train(config=None):
    logging.basicConfig(level=logging.INFO)
    discrete = False

    if config is None:
        config = SimpleNamespace(device_token=None)

    if config.device_token is None:
        device_token = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device_token = config.device_token
    device = torch.device(device_token)

    input_dim = 1
    batch_size = 1
    num_ensemble_members = 5
    ensemble = MLPEnsemble(input_dim, 1, num_ensemble_members, discrete=discrete).to(device)
    optimizer = Adam(ensemble.parameters())

    model_in = torch.randn(batch_size, input_dim).to(device)
    ensemble_out = ensemble(model_in)

    target = torch.randn(batch_size, input_dim).to(device)

    total_loss = gauss_nll_ensemble_loss(ensemble_out, target)
    optimizer.zero_grad()
    total_loss.backward()
    optimizer.step()

    print("")


if __name__ == "__main__":
    train()
    # main()
