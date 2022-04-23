import logging
import math
from enum import Enum
from typing import Tuple, Any, Dict

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.distributions import Categorical, Normal

from common import clip_mean_std
from envs.common import RewardFuncEnv

logger = logging.getLogger(__name__)


class EnsembleMode(Enum):
    ALL_MEMBERS = 1
    RANDOM_MEMBER = 2
    FIXED_MEMBER = 3
    SHUFFLED_MEMBER = 4


def get_output_shape(layer, shape):
    layer_training = layer.training
    if layer_training:
        layer.eval()
    out = layer(torch.zeros(1, *shape))
    before_flattening = tuple(out.size())[1:]
    after_flattening = int(np.prod(out.size()))
    if layer_training:
        layer.train()
    return before_flattening, after_flattening


def get_activation(activation):
    if activation == "relu":
        return nn.ReLU(inplace=True)
    elif activation == "elu":
        return nn.ELU(inplace=True)
    raise ValueError(f"Unknown Activation {activation}")


class TransitionModel(nn.Module):

    @property
    def action_dimension(self) -> int:
        raise NotImplementedError()

    @property
    def is_discrete(self) -> bool:
        raise NotImplementedError()

    @property
    def action_limits(self):
        return None

    @staticmethod
    def categorical_action_selector(policy_out, action_limits=None):
        probs = F.softmax(policy_out, dim=1)
        actions = Categorical(probs.detach().cpu()).sample().detach().cpu().numpy()
        return actions if action_limits is None else np.clip(actions, action_limits[0], action_limits[1])

    @staticmethod
    def normal_action_selector(policy_out, action_limits=None):
        mean, log_std = policy_out

        if action_limits is not None:
            low, high = action_limits
            mean, log_std = clip_mean_std(mean, log_std, low, high)

        std_dev = torch.exp(log_std)
        actions = Normal(mean, std_dev).sample().detach().cpu().numpy()
        return actions if action_limits is None else np.clip(actions, action_limits[0], action_limits[1])

    def select_actions(self, policy_out, action_limits=None):
        if self.is_discrete:
            return self.categorical_action_selector(policy_out, action_limits)
        return self.normal_action_selector(policy_out, action_limits)


class PolicyEnsemble(object):

    @property
    def num_members(self) -> int:
        raise NotImplementedError()

    def train(self):
        raise NotImplementedError()

    def eval(self):
        raise NotImplementedError()

    def shuffle_ids(self, batch_size):
        raise NotImplementedError()

    def forward(self, *args, **kwargs):
        raise NotImplementedError()

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)


class MLPModel(TransitionModel):

    def __init__(self, state_dimension, action_dimension, discrete=False, fully_params=None, activation=None):
        super().__init__()

        if fully_params is None:
            fully_params = [64, 64]
        if activation is None:
            activation = "relu"

        self.action_dim = action_dimension
        self.state_dimension = state_dimension
        self.activation = activation
        self.discrete = discrete

        layers = []
        prev_full_n = self.state_dimension + self.action_dim

        for full_n in fully_params:
            layers.append(nn.Linear(prev_full_n, full_n))
            layers.append(get_activation(self.activation))
            prev_full_n = full_n

        self.shared_layers = nn.Sequential(*layers)
        self.mean = nn.Linear(prev_full_n, self.state_dimension)
        if not self.discrete:
            self.log_std = nn.Linear(prev_full_n, self.state_dimension)
        else:
            self.log_std = None

    @property
    def action_dimension(self) -> int:
        return self.action_dim

    @property
    def is_discrete(self) -> bool:
        return self.discrete

    def forward(self, x):
        if self.discrete:
            return self.mean(self.shared_layers(x))
        p_shared_out = self.shared_layers(x)
        return self.mean(p_shared_out), self.log_std(p_shared_out)


class MLPEnsemble(PolicyEnsemble):

    def __init__(self, input_size, action_dimension, num_members, discrete=False, fully_params=None, activation=None,
                 ensemble_mode=EnsembleMode.ALL_MEMBERS):
        self._num_members = num_members
        self.members = [
            MLPModel(input_size, action_dimension, discrete=discrete, fully_params=fully_params, activation=activation)
            for _ in range(num_members)]
        self.ensemble_mode = ensemble_mode
        self.permuted_ids = None
        self.reverse_permuted_ids = None

    def parameters(self):
        params = []
        for member in self.members:
            params += list(member.parameters())
        return params

    def to(self, device):
        for model in self.members:
            model.to(device)
        return self

    def train(self):
        for m in self.members:
            m.train()

    def eval(self):
        for m in self.members:
            m.eval()

    @property
    def num_members(self) -> int:
        return self._num_members

    def shuffle_ids(self, batch_size):
        self.permuted_ids = np.random.permutation(batch_size)
        self.reverse_permuted_ids = np.argsort(self.permuted_ids)

    def forward(self, x):
        if self.ensemble_mode == EnsembleMode.ALL_MEMBERS:
            if (isinstance(x, list) or isinstance(x, tuple)) and len(x) == self.num_members:
                return [model(m_in) for model, m_in in zip(self.members, x)]
            elif torch.is_tensor(x):
                return [model(x) for model in self.members]
            else:
                raise ValueError("Input needs to be tensor or list with length equal to number of members.")
        elif self.ensemble_mode == EnsembleMode.FIXED_MEMBER or self.ensemble_mode == EnsembleMode.SHUFFLED_MEMBER or self.ensemble_mode == EnsembleMode.RANDOM_MEMBER:

            if self.ensemble_mode != EnsembleMode.FIXED_MEMBER:
                batch_size = x.shape[0]
                if (self.permuted_ids is None) or self.ensemble_mode == EnsembleMode.RANDOM_MEMBER:
                    self.shuffle_ids(batch_size)
                x = x[self.permuted_ids]

            x_per_member = x.view(self.num_members, -1, *x.shape[1:])
            results = [model(el) for model, el in zip(self.members, x_per_member)]
            means, log_stds = list(zip(*results))
            mean_out = torch.cat(means)
            log_std_out = torch.cat(log_stds)

            if self.ensemble_mode != EnsembleMode.FIXED_MEMBER:
                # shuffle back
                mean_out = mean_out[self.reverse_permuted_ids]
                log_std_out = log_std_out[self.reverse_permuted_ids]
                x = x[self.reverse_permuted_ids]  # TODO remove?

            return mean_out, log_std_out

        raise NotImplementedError()


class DynamicsModel(object):

    def step(self, states, actions) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict]:
        """Run one timestep of the model's dynamics.

        Accepts an action and returns a tuple (observation, reward, done, info).

        :param states: the current state(s)
        :param actions: the action(s) taken

        :return
            next_states (object): agent's observation of the current environment
            rewards (np.ndarray) : amount of reward returned after previous action
            dones (np.ndarray): whether the episode has ended
            info (dict): contains auxiliary diagnostic information (helpful for debugging, and sometimes learning)
        """
        raise NotImplementedError()


class EnsembleDynamicsModel(DynamicsModel):

    def __init__(self, ensemble, env: RewardFuncEnv, device, data_type=torch.float32):
        self.ensemble = ensemble
        self.env = env
        self.device = device
        self.data_type = data_type

    def step(self, states, actions) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict]:
        states_t = torch.from_numpy(states).type(self.data_type).to(self.device)
        actions_t = torch.from_numpy(actions).type(self.data_type).to(self.device)

        model_in_t = torch.cat([states_t, actions_t], dim=-1)

        means, log_stds = self.ensemble(model_in_t)
        std_devs = torch.exp(log_stds)

        next_states = Normal(means, std_devs).sample().detach().cpu().numpy()

        dones = np.array([self.env.is_done(s, n_s) for s, n_s in zip(states, next_states)])
        rewards = np.array([self.env.reward(s, n_s, d) for s, n_s, d in zip(states, next_states, dones)])
        return states, rewards, dones, {}
