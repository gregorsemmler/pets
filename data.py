import numpy as np


class ModelBatch(object):

    def __init__(self, states, actions, next_states, rewards, dones):
        self.states = states
        self.actions = actions
        self.next_states = next_states
        self.values = rewards
        self.dones = dones

    def __len__(self):
        return len(self.states)


class ReplayBuffer(object):

    def __init__(self, size, state_shape, action_shape, obs_type=np.float32, action_type=np.float32,
                 reward_type=np.float32):
        self.size = size
        self.states = np.empty((size, *state_shape), dtype=obs_type)
        self.actions = np.empty((size, *action_shape), dtype=action_type)
        self.next_states = np.empty((size, *state_shape), dtype=obs_type)
        self.rewards = np.empty(size, dtype=reward_type)
        self.dones = np.empty(size, dtype=bool)

        self.cur_idx = 0
        self.num_stored = 0

    def add(self, state, action, next_state, reward, done):
        self.states[self.cur_idx] = state
        self.actions[self.cur_idx] = action
        self.next_states[self.cur_idx] = next_state
        self.rewards[self.cur_idx] = reward
        self.dones[self.cur_idx] = done

        self.cur_idx = (self.cur_idx + 1) % self.size
        self.num_stored = min(self.num_stored + 1, self.size)

    def sample(self, size):
        ids = np.random.choice(self.num_stored, size=size)
        return ModelBatch(self.states[ids], self.actions[ids], self.next_states[ids], self.rewards[ids],
                          self.dones[ids])

    def reset(self):
        self.cur_idx = 0
        self.num_stored = 0
