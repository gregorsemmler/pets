import gym
import numpy as np
import torch
from gym import envs

from envs.cartpole_continuous import ContinuousCartPoleEnv
from model import DynamicsModel, EnsembleDynamicsModel, MLPEnsemble, EnsembleMode
from optimizer import CEMOptimizer


def main():
    env_names = sorted(envs.registry.env_specs.keys())
    xx = gym.make("CartPole-v1")
    print("")


def random_agent_evaluation():
    env = ContinuousCartPoleEnv()
    device_token = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device_token)
    batch_size = 100
    num_particles = 15
    horizon = 10
    state_dim = 4
    action_dim = 1
    num_ensemble_members = 5

    env_state = env.reset()
    actions = np.clip(np.random.randn(batch_size, horizon, action_dim), -1, 1).astype(np.float32)
    # TODO test
    # actions = np.tile(np.linspace(-1, 1, batch_size)[:, np.newaxis], (1, horizon))[..., np.newaxis].astype(np.float32)
    ensemble = MLPEnsemble(state_dim, action_dim, num_ensemble_members, ensemble_mode=EnsembleMode.SHUFFLED_MEMBER).to(
        device)
    dynamics_model = EnsembleDynamicsModel(ensemble, env, device)

    print("")
    average_rewards = dynamics_model_evaluation(env_state, dynamics_model, actions, num_particles)

    print("")
    pass


def dynamics_model_evaluation(initial_state, dynamics_model: DynamicsModel, horizon_actions, num_particles):
    batch_size, horizon = horizon_actions.shape[:2]
    states = np.tile(initial_state, (batch_size * num_particles,) + tuple([1] * initial_state.ndim))
    average_rewards = np.zeros(batch_size * num_particles, dtype=np.float32)
    done_envs = np.zeros((batch_size * num_particles, ), dtype=bool)
    for time_step in range(horizon):
        action = horizon_actions[:, time_step, :]
        actions = np.tile(action, (num_particles, 1))
        states, rewards, dones, _ = dynamics_model.step(states, actions)
        rewards[done_envs] = 0.0
        done_envs |= dones
        average_rewards += rewards

    average_rewards = average_rewards.reshape(-1, num_particles).mean(axis=-1)
    return average_rewards


def cem_experiment():
    num_samples = 100
    elite_size = 20
    horizon = 1
    num_iterations = 100
    lower_bound = -2
    upper_bound = 2
    alpha = 0.1
    cem_opt = CEMOptimizer(num_samples, elite_size, horizon, num_iterations, lower_bound, upper_bound, alpha, )
    pass


if __name__ == "__main__":
    # main()
    random_agent_evaluation()
