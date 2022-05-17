import logging
import math
from enum import Enum
from typing import Tuple, Dict, Optional, Union

import gym
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.distributions import Normal

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


def get_activation(act):
    if act == "relu":
        return nn.ReLU(inplace=True)
    elif act == "elu":
        return nn.ELU(inplace=True)
    elif act == "leaky_relu":
        return nn.LeakyReLU(inplace=True)
    raise ValueError(f"Unknown Activation {act}")


class PolicyEnsemble(object):

    @property
    def predict_rewards(self) -> bool:
        raise NotImplementedError()

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

    def __init__(self, ensemble: PolicyEnsemble, env: Union[RewardFuncEnv, gym.Env], device,
                 normalizer: Optional[Normalizer] = None, done_threshold=0.5,
                 data_type=torch.float32):

        if not (isinstance(env, RewardFuncEnv) or ensemble.predict_rewards):
            raise ValueError("Either the ensemble must predict the rewards or a RewardFuncEnv must be supplied.")
        self.ensemble = ensemble
        self.env = env
        self.normalizer = normalizer
        self.device = device
        self.done_threshold = done_threshold
        self.data_type = data_type

    def step(self, states, actions) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict]:
        states_t = torch.from_numpy(states).type(self.data_type).to(self.device)
        actions_t = torch.from_numpy(actions).type(self.data_type).to(self.device)

        model_in_t = torch.cat([states_t, actions_t], dim=-1)

        if self.normalizer is not None:
            model_in_t = self.normalizer.normalize(model_in_t)

        if not self.ensemble.predict_rewards:
            means, log_stds = self.ensemble(model_in_t)
            rewards, done_out = None, None
        else:
            means, log_stds, rewards, done_out = self.ensemble(model_in_t)

        std_devs = torch.exp(log_stds)

        next_states = Normal(means, std_devs).sample().detach().cpu().numpy()

        if rewards is None:
            dones = self.env.is_done(states, next_states)
            rewards = self.env.reward(states, next_states, dones)
        else:
            done_scores = torch.sigmoid(done_out)
            done_scores = done_scores.detach().cpu().numpy()
            dones = (done_scores > self.done_threshold).squeeze()
            rewards = rewards.detach().cpu().numpy().squeeze()

        return next_states, rewards, dones, {}


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


class MLPEnsemble(nn.Module, PolicyEnsemble):

    def __init__(self, state_dimension, action_dimension, num_members, discrete=False, fully_params=None,
                 reward_params=None, activation=None, ensemble_mode=EnsembleMode.ALL_MEMBERS, min_log_std=-5.0,
                 max_log_std=0.25):
        super().__init__()

        if fully_params is None:
            fully_params = [64, 64]
        if activation is None:
            activation = "relu"

        self._predict_rewards = reward_params is not None
        self.action_dim = action_dimension
        self.state_dimension = state_dimension
        self.activation = activation
        self.discrete = discrete
        self._num_members = num_members

        layers = []
        prev_full_n = self.state_dimension + self.action_dim

        for full_n in fully_params:
            layers.append(EnsembleLinear(prev_full_n, full_n, self._num_members))
            layers.append(get_activation(self.activation))
            prev_full_n = full_n

        self.shared_layers = nn.Sequential(*layers)
        if not self.discrete:
            self.mean_and_log_std = nn.Linear(prev_full_n,  2*self.state_dimension)
        else:
            self.mean_and_log_std = nn.Linear(prev_full_n,  self.state_dimension)

        if self._predict_rewards:
            rew_layers = []
            prev_reward_n = prev_full_n

            for reward_n in reward_params:
                rew_layers.append(EnsembleLinear(prev_reward_n, reward_n, self._num_members))
                rew_layers.append(get_activation(self.activation))
                prev_reward_n = reward_n
            rew_layers.append(nn.Linear(prev_reward_n, 2))

            self.reward_layers = nn.Sequential(*rew_layers)

        self.ensemble_mode = ensemble_mode
        self.permuted_ids = None
        self.reverse_permuted_ids = None
        self._min_log_std = nn.Parameter(min_log_std * torch.ones(state_dimension))
        self._max_log_std = nn.Parameter(max_log_std * torch.ones(state_dimension))

    @property
    def predict_rewards(self) -> bool:
        return self._predict_rewards

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

    def ensemble_forward(self, x):
        shared_out = self.shared_layers(x)
        m_and_log_s = self.mean_and_log_std(shared_out)
        mean, log_std = m_and_log_s[..., :self.state_dimension], m_and_log_s[..., self.state_dimension:]
        log_std = self.limit_log_std(log_std)
        if not self._predict_rewards:
            return mean, log_std
        rewards_and_done_out = self.reward_layers(shared_out)
        reward, done_out = rewards_and_done_out[..., :1], rewards_and_done_out[..., 1:]
        return mean, log_std, reward, done_out

    def forward(self, x):

        if self.ensemble_mode == EnsembleMode.ALL_MEMBERS:
            if (isinstance(x, list) or isinstance(x, tuple)) and len(x) == self.num_members:
                x = torch.stack(x)
            if torch.is_tensor(x):
                return self.ensemble_forward(x)
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
            if not self._predict_rewards:
                mean_out, log_std_out = self.ensemble_forward(x_per_member)
                reward, done_out = None, None
            else:
                mean_out, log_std_out, reward, done_out = self.ensemble_forward(x_per_member)

            mean_out = mean_out.view(-1, mean_out.shape[-1])
            log_std_out = log_std_out.view(-1, mean_out.shape[-1])
            if self._predict_rewards:
                reward = reward.view(-1, reward.shape[-1])
                done_out = done_out.view(-1, done_out.shape[-1])

            if self.ensemble_mode != EnsembleMode.FIXED_MEMBER:
                # shuffle back
                mean_out = mean_out[self.reverse_permuted_ids]
                log_std_out = log_std_out[self.reverse_permuted_ids]

                if self._predict_rewards:
                    reward = reward[self.reverse_permuted_ids]
                    done_out = done_out[self.reverse_permuted_ids]

            if not self._predict_rewards:
                return mean_out, log_std_out
            return mean_out, log_std_out, reward, done_out

        raise NotImplementedError()
