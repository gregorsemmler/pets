import gym
import numpy as np


class RewardFuncEnv(gym.Env):

    def step(self, action):
        raise NotImplementedError()

    def reset(self):
        raise NotImplementedError()

    def render(self, mode="human"):
        raise NotImplementedError()

    def reward(self, state, new_state, done) -> np.ndarray:
        """
        Returns the reward obtained when transitioning from a state into a new state, depending on whether the
        environment is done or not.
        :param state: the current state.
        :param new_state: the next state.
        :param done: a boolean indicating if the environment is done.
        :return: the reward obtained from this transition.
        """
        raise NotImplementedError()

    def is_done(self, state, new_state) -> np.ndarray:
        """
        Returns whether the environment is done when transitioning from one state to the next.
        :param state: the current state.
        :param new_state: the next state.
        :return: a bool indicating whether or not the environment is done.
        """
        raise NotImplementedError()
