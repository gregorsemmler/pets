import gym
import numpy as np
from gym import envs

from envs.cartpole_continuous import ContinuousCartPoleEnv
from optimizer import CEMOptimizer


def main():
    env_names = sorted(envs.registry.env_specs.keys())
    xx = gym.make("CartPole-v1")
    print("")


def random_agent_evaluation():
    env = ContinuousCartPoleEnv()
    num_trials = 10
    for trial_id in range(num_trials):
        print(f"Trial {trial_id}")
        env.reset()
        done = False
        while not done:
            env.render()
            action = np.clip(np.random.randn(1), -1, 1).astype(np.float32)
            state, reward, done, _ = env.step(action)

    print("")
    pass


def environment_evaluation(env, actions, num_particles):
    # TODO input: batch of action sequences(, num particles) Output: Average total rewards for each batch element
    pass


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
