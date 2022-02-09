import logging
import math

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.distributions import Categorical, Normal

from common import clip_mean_std

logger = logging.getLogger(__name__)


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


class PolicyModel(nn.Module):

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

    def forward(self, *args, **kwargs):
        raise NotImplementedError()

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)


class MLPModel(PolicyModel):

    def __init__(self, input_size, action_dimension, discrete=False, fully_params=None, activation=None):
        super().__init__()

        if fully_params is None:
            fully_params = [64, 64]
        if activation is None:
            activation = "relu"

        self.action_dim = action_dimension
        self.input_size = input_size
        self.activation = activation
        self.discrete = discrete

        layers = []
        prev_full_n = self.input_size

        for full_n in fully_params:
            layers.append(nn.Linear(prev_full_n, full_n))
            layers.append(get_activation(self.activation))
            prev_full_n = full_n

        self.shared_layers = nn.Sequential(*layers)
        self.mean = nn.Linear(prev_full_n, action_dimension)
        if not self.discrete:
            self.log_std = nn.Linear(prev_full_n, action_dimension)
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

    def __init__(self, input_size, action_dimension, num_members, discrete=False, fully_params=None, activation=None):
        super().__init__()

        self.members = [
            MLPModel(input_size, action_dimension, discrete=discrete, fully_params=fully_params, activation=activation)
            for _ in range(num_members)]

    def parameters(self):
        params = []
        for member in self.members:
            params += list(member.parameters())
        return params

    def to(self, device):
        for model in self.members:
            model.to(device)
        return self

    @property
    def num_members(self) -> int:
        return len(self.members)

    def forward(self, x):
        return [model(x) for model in self.members]
