import logging
from math import pi
from types import SimpleNamespace

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from matplotlib import cm
from torch import nn
from torch.optim import Adam

from data import ReplayBuffer, SimpleBatchProcessor
from model import MLPEnsemble
from optimizer import CEMOptimizer


def generate_simple_dataset(x_low=-2 * pi, x_high=2 * pi, data_size=10000, val_ratio=0.1, train_size=0.8,
                            noise_strength=0.15, noise_type=1):

    train_limits = int(data_size * train_size * 0.5)

    orig_x = np.linspace(x_low, x_high, data_size)
    orig_y = np.sin(orig_x)

    x = np.concatenate([orig_x[:train_limits], orig_x[-train_limits:]])
    no_noise_y = np.sin(x)

    noise1 = np.random.randn(*x.shape) * np.sqrt(np.abs(np.sin(1.5 * x + pi / 8)))
    noise2 = np.random.randn(*x.shape)

    y1 = no_noise_y + noise_strength * noise1
    y2 = no_noise_y + noise_strength * noise2

    if noise_type == 1:
        y = y1
    elif noise_type == 2:
        y = y2
    else:
        raise ValueError(f"Unknown noise_type '{noise_type}'")

    val_size = int(val_ratio * len(x))
    shuffled_ids = np.random.permutation(len(x))
    train_ids = shuffled_ids[val_size:]
    val_ids = shuffled_ids[:val_size]

    train_buffer = ReplayBuffer(data_size, (1,), (0,))
    val_buffer = ReplayBuffer(data_size, (1,), (0,))

    for train_id in train_ids:
        train_buffer.add(x[train_id], np.array([0]), y[train_id], 0, False)
    for val_id in val_ids:
        val_buffer.add(x[val_id], np.array([0]), y[val_id], 0, False)

    return orig_x, orig_y, x, y, train_buffer, val_buffer


def gauss_nll_ensemble_loss(ensemble_out, targets):
    losses = []
    if len(targets) != len(ensemble_out):
        raise ValueError("Model output is not same length as target")

    for (mean_model, log_std_model), target in zip(ensemble_out, targets):
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
    batch_size = 128
    num_epochs = 100
    num_ensemble_members = 5
    ensemble = MLPEnsemble(input_dim, 0, num_ensemble_members, discrete=discrete).to(device)
    optimizer = Adam(ensemble.parameters())

    print("")

    orig_x, orig_y, x, y, train_buffer, val_buffer = generate_simple_dataset(train_size=0.7, noise_strength=0.3,
                                                                             noise_type=1)
    processor = SimpleBatchProcessor(device)

    batch_idx = 0
    log_frequency = 100

    for epoch_idx in range(num_epochs):
        for ensemble_batches in train_buffer.batches(batch_size, num_ensemble_members):
            model_in, target_out = list(zip(*[processor.process(b) for b in ensemble_batches]))
            model_out = ensemble(model_in)

            total_loss = gauss_nll_ensemble_loss(model_out, target_out)
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

            if batch_idx % log_frequency == 0:
                print(f"Epoch #{epoch_idx} Batch #{batch_idx} Loss: {total_loss.item()}")

            batch_idx += 1

    print("")
    ensemble_train_data = []
    for ensemble_id in range(num_ensemble_members):
        ensemble_rep_ids = sorted(set(train_buffer.batch_indices[ensemble_id]))
        xs = train_buffer.states[ensemble_rep_ids]
        ys = train_buffer.next_states[ensemble_rep_ids]
        ensemble_train_data.append((xs, ys))

    x_t = torch.from_numpy(orig_x[:, np.newaxis]).type(torch.float32).to(device)

    with torch.no_grad():
        model_out = ensemble([x_t for _ in range(num_ensemble_members)])
        means, log_stds = list(zip(*model_out))
        np_means = [m.cpu().numpy().squeeze() for m in means]
        np_stds = [np.exp(log_s.cpu().numpy()).squeeze() for log_s in log_stds]
        print("")

    pred_y = np.stack(np_means).mean(axis=0)
    ensemble_mean_std = np.stack(np_means).std(axis=0)
    ensemble_std = np.stack(np_stds).mean(axis=0)
    std_average = np.stack([ensemble_mean_std, ensemble_std]).mean(axis=0)

    plt.figure(figsize=(16, 12), dpi=100)
    plt.plot(orig_x, orig_y, "b", orig_x, pred_y, "r")
    plt.plot(x, y, ".g", alpha=0.2)
    plt.fill_between(orig_x, pred_y - std_average, pred_y + std_average, color="r", alpha=0.2)
    # plt.fill_between(orig_x, pred_y - ensemble_std, pred_y + ensemble_std, color="k", alpha=0.2)
    # plt.fill_between(orig_x, pred_y - ensemble_mean_std, pred_y + ensemble_mean_std, color="m", alpha=0.2)
    plt.show()

    print("")
    cm_name = "Dark2"

    cmap = cm.get_cmap(cm_name)
    plt.figure(figsize=(16, 12), dpi=100)
    plt.plot(orig_x, orig_y, "b")

    e_alpha = 0.2
    for e_id, ((e_train_x, e_train_y), m, s) in enumerate(zip(ensemble_train_data, np_means, np_stds)):
        # plt.plot(e_train_x, e_train_y, ".", alpha=e_alpha, color=cmap(e_id))
        plt.plot(orig_x, m, color=cmap(e_id))
        plt.fill_between(orig_x, m - s, m + s, alpha=e_alpha, color=cmap(e_id))
    plt.show()

    print("")

    cmap = cm.get_cmap(cm_name)
    e_alpha = 0.3
    for e_id, ((e_train_x, e_train_y), m, s) in enumerate(zip(ensemble_train_data, np_means, np_stds)):
        plt.figure(figsize=(16, 12), dpi=100)
        plt.plot(orig_x, orig_y, "b")
        plt.plot(e_train_x, e_train_y, ".", alpha=e_alpha, color=cmap(e_id))
        plt.plot(orig_x, m, color=cmap(e_id))
        plt.fill_between(orig_x, m - s, m + s, alpha=e_alpha, color=cmap(e_id))
        plt.show()

    print("")


if __name__ == "__main__":
    train()
    # main()
