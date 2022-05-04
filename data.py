from typing import Optional

import numpy as np
import torch
from torch.nn import functional as F


class ModelBatch(object):

    def __init__(self, states, actions, next_states, rewards, dones):
        self.states = states
        self.actions = actions
        self.next_states = next_states
        self.rewards = rewards
        self.dones = dones

    def __len__(self):
        return len(self.states)


class Normalizer(object):

    def normalize(self, x: torch.tensor) -> torch.tensor:
        raise NotImplementedError()


class StandardNormalizer(Normalizer):

    def __init__(self, data: np.ndarray, device, axis=0, dtype=torch.float32, eps=1e-8):
        mean, std = data.mean(axis=axis), data.std(axis=axis)
        self.mean = torch.tensor(mean).to(device).type(dtype)
        self.std = torch.tensor(std).to(device).to(dtype)

        self.device = device
        self.dtype = dtype
        self.eps = eps

        if self.eps is not None:
            self.std[self.std < self.eps] = 1.0

    def normalize(self, x: torch.tensor) -> torch.tensor:
        return (x - self.mean) / self.std


class BatchProcessor(object):

    def process(self, batch: ModelBatch):
        raise NotImplementedError()


class SimpleBatchProcessor(BatchProcessor):

    def __init__(self, device, normalizer: Optional[Normalizer] = None, data_type=torch.float32):
        self.device = device
        self.normalizer = normalizer
        self.data_type = data_type

    def process(self, batch: ModelBatch):
        states_t = torch.from_numpy(batch.states).type(self.data_type).to(self.device)
        actions_t = torch.from_numpy(batch.actions).type(self.data_type).to(self.device)
        next_states_t = torch.from_numpy(batch.next_states).type(self.data_type).to(self.device)

        model_in_t = torch.cat([states_t, actions_t], dim=-1)

        if self.normalizer is not None:
            model_in_t = self.normalizer.normalize(model_in_t)

        return model_in_t, next_states_t


class ReplayBuffer(object):

    def __init__(self, size, state_shape, action_shape, state_type=np.float32, action_type=np.float32,
                 reward_type=np.float32):
        self.size = size
        self.states = np.empty((size, *state_shape), dtype=state_type)
        self.actions = np.empty((size, *action_shape), dtype=action_type)
        self.next_states = np.empty((size, *state_shape), dtype=state_type)
        self.rewards = np.empty(size, dtype=reward_type)
        self.dones = np.empty(size, dtype=bool)

        self.cur_idx = 0
        self.num_stored = 0
        self.batch_indices = None

    def __len__(self):
        return self.num_stored

    @classmethod
    def from_data(cls, states: np.ndarray, actions: np.ndarray, next_states: np.ndarray, rewards: np.ndarray,
                  dones: np.ndarray):
        if len({len(el) for el in [states, actions, next_states, rewards, dones]}) != 1:
            raise ValueError("Lengths need to be equal.")

        size = len(states)
        if size == 0:
            raise ValueError("Need to have at least one element.")

        state_shape = states[0].shape
        action_shape = actions[0].shape

        if state_shape != next_states[0].shape:
            raise ValueError("State shapes do not match.")

        if states.dtype != next_states.dtype:
            raise ValueError("State data types do not match.")

        result = ReplayBuffer(size, state_shape, action_shape, states.dtype, actions.dtype, rewards.dtype)
        result.states = states
        result.actions = actions
        result.next_states = next_states
        result.rewards = rewards
        result.dones = dones
        result.cur_idx = size - 1
        result.num_stored = size
        return result

    def add(self, state, action, next_state, reward, done):
        self.states[self.cur_idx] = state
        self.actions[self.cur_idx] = action
        self.next_states[self.cur_idx] = next_state
        self.rewards[self.cur_idx] = reward
        self.dones[self.cur_idx] = done

        self.cur_idx = (self.cur_idx + 1) % self.size
        self.num_stored = min(self.num_stored + 1, self.size)

    def batch_from_ids(self, ids):
        return ModelBatch(self.states[ids], self.actions[ids], self.next_states[ids], self.rewards[ids],
                          self.dones[ids])

    def batches(self, batch_size, num_ensemble_members=1, replace=True):
        self.batch_indices = np.empty((num_ensemble_members, self.num_stored), dtype=np.int64)
        for idx in range(num_ensemble_members):
            self.batch_indices[idx] = np.random.choice(self.num_stored, (self.num_stored,), replace=replace)

        start_idx = 0

        while start_idx < self.num_stored:
            end_idx = min(start_idx + batch_size, self.num_stored)
            model_batches = [self.batch_from_ids(self.batch_indices[ensemble_id, start_idx:end_idx]) for ensemble_id in
                             range(num_ensemble_members)]

            yield model_batches
            start_idx = end_idx

    def train_val_split(self, val_ratio=0.1, shuffle=True):
        states_c = self.states[:self.num_stored, ...].copy()
        actions_c = self.actions[:self.num_stored, ...].copy()
        next_states_c = self.next_states[:self.num_stored, ...].copy()
        rewards_c = self.rewards[:self.num_stored, ...].copy()
        dones_c = self.dones[:self.num_stored, ...].copy()

        val_size = int(val_ratio * self.num_stored)

        if shuffle:
            permuted_ids = np.random.permutation(self.num_stored)
            states_c = states_c[permuted_ids]
            actions_c = actions_c[permuted_ids]
            next_states_c = next_states_c[permuted_ids]
            rewards_c = rewards_c[permuted_ids]
            dones_c = dones_c[permuted_ids]

        val_states, train_states = states_c[:val_size], states_c[val_size:]
        val_actions, train_actions = actions_c[:val_size], actions_c[val_size:]
        val_next_states, train_next_states = next_states_c[:val_size], next_states_c[val_size:]
        val_rewards, train_rewards = rewards_c[:val_size], rewards_c[val_size:]
        val_dones, train_dones = dones_c[:val_size], dones_c[val_size:]

        train_buffer = ReplayBuffer.from_data(train_states, train_actions, train_next_states, train_rewards, train_dones)
        val_buffer = ReplayBuffer.from_data(val_states, val_actions, val_next_states, val_rewards, val_dones)
        return train_buffer, val_buffer

    def reset(self):
        self.cur_idx = 0
        self.num_stored = 0
        self.batch_indices = None


def gauss_nll_ensemble_loss(ensemble_out, targets):
    losses = []
    if isinstance(ensemble_out, tuple) and len(ensemble_out) == 2:
        ensemble_means, ensemble_log_stds = ensemble_out
    elif len(targets) == len(ensemble_out):
        ensemble_means, ensemble_log_stds = tuple(zip(*ensemble_out))
    else:
        raise ValueError("Wrong Model output format")

    for mean_model, log_std_model, target in zip(ensemble_means, ensemble_log_stds, targets):
        var = torch.exp(2 * log_std_model)
        model_loss = F.gaussian_nll_loss(mean_model, target, var)
        losses.append(model_loss)

    total_loss = torch.stack(losses).mean()
    return total_loss
