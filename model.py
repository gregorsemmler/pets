import logging
import math
from enum import Enum
from typing import Tuple, Any, Dict, Optional

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.distributions import Categorical, Normal

from common import clip_mean_std
from data import Normalizer
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

    @property
    def min_log_std(self):
        raise NotImplementedError()

    @property
    def max_log_std(self):
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

    def __init__(self, state_dimension, action_dimension, num_members, discrete=False, fully_params=None,
                 activation=None, ensemble_mode=EnsembleMode.ALL_MEMBERS, min_log_std=-5.0, max_log_std=0.25):
        self._num_members = num_members
        self.members = [
            MLPModel(state_dimension, action_dimension, discrete=discrete, fully_params=fully_params, activation=activation)
            for _ in range(num_members)]
        self.ensemble_mode = ensemble_mode
        self.permuted_ids = None
        self.reverse_permuted_ids = None
        self._min_log_std = nn.Parameter(min_log_std * torch.ones(state_dimension))
        self._max_log_std = nn.Parameter(max_log_std * torch.ones(state_dimension))

    def parameters(self):
        params = []
        for member in self.members:
            params += list(member.parameters())
        params += [self._min_log_std, self._max_log_std]
        return params

    def to(self, device):
        for model in self.members:
            model.to(device)
        self._min_log_std = nn.Parameter(
            torch.tensor(self._min_log_std.detach().cpu().numpy(), device=device, requires_grad=True))
        self._max_log_std = nn.Parameter(
            torch.tensor(self._max_log_std.detach().cpu().numpy(), device=device, requires_grad=True))
        return self

    def train(self):
        for m in self.members:
            m.train()

    def eval(self):
        for m in self.members:
            m.eval()

    @property
    def min_log_std(self):
        return self._min_log_std

    @property
    def max_log_std(self):
        return self._max_log_std

    @property
    def num_members(self) -> int:
        return self._num_members

    def shuffle_ids(self, batch_size):
        self.permuted_ids = np.random.permutation(batch_size)
        self.reverse_permuted_ids = np.argsort(self.permuted_ids)

    def limit_log_std(self, log_std):
        # https://github.com/kchua/handful-of-trials/blob/77fd8802cc30b7683f0227c90527b5414c0df34c/dmbrl/modeling/models/BNN.py#L414-L415
        log_std = self._max_log_std - F.softplus(self._max_log_std - log_std)
        log_std = self._min_log_std + F.softplus(log_std - self._min_log_std)
        return log_std

    def forward(self, x):
        if self.ensemble_mode == EnsembleMode.ALL_MEMBERS:
            if (isinstance(x, list) or isinstance(x, tuple)) and len(x) == self.num_members:
                result = [model(m_in) for model, m_in in zip(self.members, x)]
                result = [(mean, self.limit_log_std(log_std)) for mean, log_std in result]
                return result
            elif torch.is_tensor(x):
                result = [model(x) for model in self.members]
                result = [(mean, self.limit_log_std(log_std)) for mean, log_std in result]
                return result
            else:
                raise ValueError("Input needs to be tensor or list with length equal to number of members.")
        elif self.ensemble_mode == EnsembleMode.FIXED_MEMBER or self.ensemble_mode == EnsembleMode.SHUFFLED_MEMBER \
                or self.ensemble_mode == EnsembleMode.RANDOM_MEMBER:

            if self.ensemble_mode != EnsembleMode.FIXED_MEMBER:
                batch_size = x.shape[0]
                if (self.permuted_ids is None) or self.ensemble_mode == EnsembleMode.RANDOM_MEMBER:
                    self.shuffle_ids(batch_size)
                x = x[self.permuted_ids]

            x_per_member = x.view(self.num_members, -1, *x.shape[1:])
            results = [model(el) for model, el in zip(self.members, x_per_member)]
            means, log_stds = list(zip(*results))
            log_stds = [self.limit_log_std(el) for el in log_stds]
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

    def __init__(self, ensemble, env: RewardFuncEnv, device, normalizer: Optional[Normalizer] = None,
                 data_type=torch.float32):
        self.ensemble = ensemble
        self.env = env
        self.normalizer = normalizer
        self.device = device
        self.data_type = data_type

    def step(self, states, actions) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict]:
        states_t = torch.from_numpy(states).type(self.data_type).to(self.device)
        actions_t = torch.from_numpy(actions).type(self.data_type).to(self.device)

        model_in_t = torch.cat([states_t, actions_t], dim=-1)

        if self.normalizer is not None:
            model_in_t = self.normalizer.normalize(model_in_t)

        means, log_stds = self.ensemble(model_in_t)
        std_devs = torch.exp(log_stds)

        next_states = Normal(means, std_devs).sample().detach().cpu().numpy()

        dones = np.array([self.env.is_done(s, n_s) for s, n_s in zip(states, next_states)])
        rewards = np.array([self.env.reward(s, n_s, d) for s, n_s, d in zip(states, next_states, dones)])
        return states, rewards, dones, {}


class EnsembleLinear(nn.Module):
    """
    A Linear Layer for an Ensemble
    """

    def __init__(self, in_features, out_features, num_members=1, bias=True, device=None, dtype=None):
        super().__init__()
        kwargs = {"device": device, "dtype": dtype}
        self.in_features = in_features
        self.out_features = out_features
        self.num_members = num_members
        self.weight = nn.Parameter(torch.empty((num_members, in_features, out_features), **kwargs))
        if bias:
            self.bias = nn.Parameter(torch.empty((num_members, 1, out_features), **kwargs))
        else:
            self.register_parameter("bias", None)
            self.bias = None
        self.init_parameters()

    def init_parameters(self):
        # Use default initialization from nn.Linear
        # https://github.com/pytorch/pytorch/blob/master/torch/nn/modules/linear.py#L92
        for i in range(self.num_members):
            torch.nn.init.kaiming_uniform_(self.weight[i], a=math.sqrt(5))
            if self.bias is not None:
                fan_in, _ = torch.nn.init._calculate_fan_in_and_fan_out(self.weight[i])
                bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
                torch.nn.init.uniform_(self.bias[i], -bound, bound)

    def forward(self, x):
        result = x.matmul(self.weight)
        if self.bias is not None:
            result += self.bias
        return result


class MLPEnsemble2(nn.Module, PolicyEnsemble):
    """
    More efficient implementation of Ensemble
    """

    def __init__(self, state_dimension, action_dimension, num_members, discrete=False, fully_params=None,
                 activation=None, ensemble_mode=EnsembleMode.ALL_MEMBERS, min_log_std=-5.0, max_log_std=0.25):
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
            layers.append(EnsembleLinear(prev_full_n, full_n, self.num_members))
            layers.append(get_activation(self.activation))
            prev_full_n = full_n

        self.shared_layers = nn.Sequential(*layers)
        self.mean = nn.Linear(prev_full_n, self.state_dimension)
        if not self.discrete:
            self.log_std = nn.Linear(prev_full_n, self.state_dimension)
        else:
            self.log_std = None

        self._num_members = num_members
        self.ensemble_mode = ensemble_mode
        self.permuted_ids = None
        self.reverse_permuted_ids = None
        self._min_log_std = nn.Parameter(min_log_std * torch.ones(state_dimension))
        self._max_log_std = nn.Parameter(max_log_std * torch.ones(state_dimension))

    @property
    def min_log_std(self):
        return self._min_log_std

    @property
    def max_log_std(self):
        return self._max_log_std

    @property
    def num_members(self) -> int:
        return self._num_members

    def shuffle_ids(self, batch_size):
        self.permuted_ids = np.random.permutation(batch_size)
        self.reverse_permuted_ids = np.argsort(self.permuted_ids)

    def limit_log_std(self, log_std):
        # https://github.com/kchua/handful-of-trials/blob/77fd8802cc30b7683f0227c90527b5414c0df34c/dmbrl/modeling/models/BNN.py#L414-L415
        log_std = self._max_log_std - F.softplus(self._max_log_std - log_std)
        log_std = self._min_log_std + F.softplus(log_std - self._min_log_std)
        return log_std

    def forward(self, x):
        # TODO
        if self.ensemble_mode == EnsembleMode.ALL_MEMBERS:
            if (isinstance(x, list) or isinstance(x, tuple)) and len(x) == self.num_members:
                result = [model(m_in) for model, m_in in zip(self.members, x)]
                result = [(mean, self.limit_log_std(log_std)) for mean, log_std in result]
                return result
            elif torch.is_tensor(x):
                result = [model(x) for model in self.members]
                result = [(mean, self.limit_log_std(log_std)) for mean, log_std in result]
                return result
            else:
                raise ValueError("Input needs to be tensor or list with length equal to number of members.")
        elif self.ensemble_mode == EnsembleMode.FIXED_MEMBER or self.ensemble_mode == EnsembleMode.SHUFFLED_MEMBER \
                or self.ensemble_mode == EnsembleMode.RANDOM_MEMBER:

            if self.ensemble_mode != EnsembleMode.FIXED_MEMBER:
                batch_size = x.shape[0]
                if (self.permuted_ids is None) or self.ensemble_mode == EnsembleMode.RANDOM_MEMBER:
                    self.shuffle_ids(batch_size)
                x = x[self.permuted_ids]

            x_per_member = x.view(self.num_members, -1, *x.shape[1:])
            results = [model(el) for model, el in zip(self.members, x_per_member)]
            means, log_stds = list(zip(*results))
            log_stds = [self.limit_log_std(el) for el in log_stds]
            mean_out = torch.cat(means)
            log_std_out = torch.cat(log_stds)

            if self.ensemble_mode != EnsembleMode.FIXED_MEMBER:
                # shuffle back
                mean_out = mean_out[self.reverse_permuted_ids]
                log_std_out = log_std_out[self.reverse_permuted_ids]
                x = x[self.reverse_permuted_ids]  # TODO remove?

            return mean_out, log_std_out

        raise NotImplementedError()


if __name__ == "__main__":
    fully_params = [200, 200, 200]
    layers = []
    state_dimension = 4
    action_dim = 1
    batch_size = 128
    activation = "relu"
    num_members = 5
    prev_full_n = state_dimension + action_dim
    in_size = state_dimension + action_dim

    for full_n in fully_params:
        layers.append(EnsembleLinear(prev_full_n, full_n, num_members))
        layers.append(get_activation(activation))
        prev_full_n = full_n

    shared_layers = nn.Sequential(*layers)

    print("")
    model_in = torch.randn(num_members, batch_size, in_size)
    model_out = shared_layers(model_in)
    print("")
    pass
