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


class Preprocessor(object):
    pass


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

    def batch_from_ids(self, ids):
        return ModelBatch(self.states[ids], self.actions[ids], self.next_states[ids], self.rewards[ids],
                          self.dones[ids])

    def batches(self, batch_size, num_ensemble_members=1, replace=True):
        indices = np.empty((num_ensemble_members, self.num_stored), dtype=np.int64)
        for idx in range(num_ensemble_members):
            indices[idx] = np.random.choice(self.num_stored, (self.num_stored,), replace=replace)

        start_idx = 0

        while start_idx < self.num_stored:
            end_idx = min(start_idx + batch_size, self.num_stored)
            model_batches = [self.batch_from_ids(indices[ensemble_id, start_idx:end_idx]) for ensemble_id in
                             range(num_ensemble_members)]

            yield model_batches
            start_idx = end_idx

    def reset(self):
        self.cur_idx = 0
        self.num_stored = 0
