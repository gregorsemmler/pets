import json
import logging
import math
import signal

import numpy as np
import torch
from gym.spaces import Discrete, Box


logger = logging.getLogger(__name__)


CHECKPOINT_MODEL = "model"
CHECKPOINT_OPTIMIZER = "optimizer"
CHECKPOINT_CRITIC_OPTIMIZER = "critic_optimizer"
CHECKPOINT_MODEL_ID = "model_id"


def load_json(path, *args, **kwargs):
    with open(path, "r") as f:
        return json.load(f, *args, **kwargs)


def save_json(path, data, prettify=True, *args, **kwargs):
    with open(path, "w") as f:
        if prettify:
            json.dump(data, f, indent=4, sort_keys=True, separators=(",", ": "), ensure_ascii=False, *args, **kwargs)
        else:
            json.dump(data, f, *args, **kwargs)


def load_checkpoint(path, model, optimizer=None, critic_optimizer=None, device="cpu"):
    state = torch.load(path, map_location=device)
    model.load_state_dict(state[CHECKPOINT_MODEL])
    if optimizer is not None:
        optimizer.load_state_dict(state[CHECKPOINT_OPTIMIZER])
    if critic_optimizer is not None:
        critic_optimizer.load_state_dict(state[CHECKPOINT_CRITIC_OPTIMIZER])
    epoch = state.get(CHECKPOINT_MODEL_ID)
    return epoch


def save_checkpoint(path, model, optimizer=None, critic_optimizer=None, model_id=None):
    torch.save({
        CHECKPOINT_MODEL: model.state_dict(),
        CHECKPOINT_OPTIMIZER: optimizer.state_dict() if optimizer is not None else None,
        CHECKPOINT_CRITIC_OPTIMIZER: critic_optimizer.state_dict() if critic_optimizer is not None else None,
        CHECKPOINT_MODEL_ID: model_id,
    }, path)


def get_action_space_details(action_space):
    if isinstance(action_space, Discrete):
        discrete = True
    elif isinstance(action_space, Box):
        discrete = False
    else:
        raise ValueError("Unknown type of action_space")

    action_dim = action_space.n if discrete else action_space.shape[0]
    limits = None if discrete else (action_space.low, action_space.high)
    return discrete, action_dim, limits


def clip_mean_std(mean, log_std, low, high, log_std_min=1e-5, log_std_max_factor=2):
    device = mean.device
    mean = torch.clamp(mean, torch.from_numpy(low).to(device), torch.from_numpy(high).to(device))
    log_std = torch.clamp(log_std, torch.FloatTensor([math.log(log_std_min)]).to(device),
                          log_std_max_factor * torch.log(torch.from_numpy(high - low)).to(device))
    return mean, log_std


class GracefulExit(object):

    def __init__(self):
        self.run = True
        signal.signal(signal.SIGINT, self.exit_gracefully)
        signal.signal(signal.SIGTERM, self.exit_gracefully)

    def exit_gracefully(self, *args):
        logger.info("Termination Signal received. Exiting gracefully")
        self.run = False


def parse_list(s):
    return [int(e) for e in s.split(",")]


def parse_list_of_lists(s):
    return [[int(el) for el in lst.split(",")] for lst in s.split(";")]


def shift_numpy_array(arr, num, fill_value=0.0):
    # https://stackoverflow.com/a/42642326
    result = np.empty_like(arr)
    if num > 0:
        result[:num] = fill_value
        result[num:] = arr[:-num]
    elif num < 0:
        result[num:] = fill_value
        result[:num] = arr[-num:]
    else:
        result[:] = arr
    return result