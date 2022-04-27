import logging
from timeit import default_timer as timer

import gym
import numpy as np
import torch
import torch.nn.functional as F
from gym import envs
from torch.optim import Adam

from data import ReplayBuffer, BatchProcessor, gauss_nll_ensemble_loss, SimpleBatchProcessor, StandardNormalizer
from envs.cartpole_continuous import ContinuousCartPoleEnv
from model import DynamicsModel, EnsembleDynamicsModel, MLPEnsemble, EnsembleMode, PolicyEnsemble
from optimizer import CEMOptimizer

logger = logging.getLogger(__name__)


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


def play_and_add_to_buffer(env, policy, replay_buffer: ReplayBuffer, num_time_steps=200):
    i = 0

    play_finished = False
    while not play_finished:
        if i >= num_time_steps:
            break

        state = env.reset()
        done = False

        while not done:
            action = policy(state)
            new_state, reward, done, info = env.step(action)

            replay_buffer.add(state, action, new_state, reward, done)
            state = new_state

            i += 1

            if i >= num_time_steps:
                play_finished = True
                break


def get_normalizer_for_replay_buffer(replay_buffer: ReplayBuffer, device, dtype=torch.float32,
                                     eps=1e-8) -> StandardNormalizer:
    states, actions = replay_buffer.states, replay_buffer.actions
    normalizer_data = np.concatenate([states, actions], axis=-1)
    result = StandardNormalizer(normalizer_data, device, dtype=dtype, eps=eps)
    return result


def train_on_replay_buffer(ensemble: PolicyEnsemble, train_buffer: ReplayBuffer, val_buffer: ReplayBuffer, optimizer,
                           processor: BatchProcessor, batch_size, num_epochs, log_frequency=100, trial_id=None):
    prev_ensemble_mode = ensemble.ensemble_mode
    ensemble.ensemble_mode = EnsembleMode.ALL_MEMBERS

    train_batch_idx = 0
    val_batch_idx = 0
    train_batch_losses, val_batch_losses = [], []
    train_epoch_losses, val_epoch_losses = [], []

    trial_id = "" if trial_id is None else f"Trial {trial_id}: "
    logger.info(f"{trial_id}Training for {num_epochs} epochs. (TrainSize / ValSize): ({len(train_buffer)}/{len(val_buffer)})")
    for epoch_idx in range(num_epochs):

        ensemble.train()
        train_epoch_loss = 0.0
        train_batch_count = 0
        logger.info(f"{trial_id}Training Epoch {epoch_idx}.")
        for ensemble_batches in train_buffer.batches(batch_size, ensemble.num_members):
            model_in, target_out = list(zip(*[processor.process(b) for b in ensemble_batches]))
            model_out = ensemble(model_in)

            total_loss = gauss_nll_ensemble_loss(model_out, target_out)
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
            loss_value = total_loss.item()

            if train_batch_idx % log_frequency == 0:
                logger.info(f"{trial_id}Train Epoch #{epoch_idx} Batch #{train_batch_idx} Loss: {loss_value}")

            train_epoch_loss += loss_value
            train_batch_losses.append(loss_value)
            train_batch_idx += 1
            train_batch_count += 1

        train_epoch_loss = 0.0 if train_batch_count == 0 else train_epoch_loss / train_batch_count
        train_epoch_losses.append(train_epoch_loss)

        logger.info(f"{trial_id}Validation Epoch {epoch_idx}.")
        ensemble.eval()
        val_epoch_loss = 0.0
        val_batch_count = 0
        for ensemble_batches in val_buffer.batches(batch_size, ensemble.num_members):
            model_in, target_out = list(zip(*[processor.process(b) for b in ensemble_batches]))
            with torch.no_grad():
                model_out = ensemble(model_in)

            total_loss = gauss_nll_ensemble_loss(model_out, target_out)
            loss_value = total_loss.item()

            if val_batch_idx % log_frequency == 0:
                logger.info(f"{trial_id}Val Epoch #{epoch_idx} Batch #{val_batch_idx} Loss: {loss_value}")

            val_epoch_loss += loss_value
            val_batch_losses.append(loss_value)
            val_batch_idx += 1
            val_batch_count += 1

        val_epoch_loss = 0.0 if val_batch_count == 0 else val_epoch_loss / val_batch_count
        val_epoch_losses.append(val_epoch_loss)

    ensemble.ensemble_mode = prev_ensemble_mode

    return train_batch_losses, val_batch_losses, train_epoch_losses, val_epoch_losses


def run_pets():
    logging.basicConfig(level=logging.INFO)

    env = ContinuousCartPoleEnv()
    state = env.reset()
    state_shape = state.shape
    action_shape = env.action_space.sample().shape
    device_token = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device_token)

    if len(state_shape) != 1 or len(action_shape) != 1:
        raise ValueError("State and action shape need to have dimension 1")

    state_dim = state_shape[0]
    action_dim = action_shape[0]

    replay_buffer_size = 3000
    replay_buffer = ReplayBuffer(replay_buffer_size, state_shape,  action_shape)

    # Train parameters
    train_batch_size = 32
    train_epochs = 50
    train_lr = 1e-3
    l2_regularization = 0
    val_ratio = 0.1
    shuffle = True
    train_log_frequency = 20

    num_random_steps = 1000
    # Fill replay buffer with initial data from random actions
    play_and_add_to_buffer(env, lambda x: env.action_space.sample(), replay_buffer, num_random_steps)

    num_ensemble_members = 5
    ensemble = MLPEnsemble(state_dim, action_dim, num_ensemble_members, ensemble_mode=EnsembleMode.SHUFFLED_MEMBER).to(
        device)
    optimizer = Adam(ensemble.parameters(), lr=train_lr, weight_decay=l2_regularization)
    dynamics_model = EnsembleDynamicsModel(ensemble, env, device)

    # CEM Options
    num_samples = 100
    elite_size = 10
    horizon = 10
    num_iterations = 100
    lower_bound_np, upper_bound_np = env.action_space.low, env.action_space.high
    alpha = 0.1
    num_particles = 15

    lower_bound = torch.tensor(np.tile(lower_bound_np, (horizon, 1)))
    upper_bound = torch.tensor(np.tile(upper_bound_np, (horizon, 1)))

    solution = (lower_bound + upper_bound) / 2

    trial_gamma = 1.0

    trial_lengths = []
    trial_returns = []
    best_trial_return = float("-inf")
    best_trial_length = 0
    best_trial_id = None

    num_trials = 100
    for trial_idx in range(num_trials):
        logger.info(f"Starting trial {trial_idx}.")
        ensemble.shuffle_ids(num_samples * num_particles)  # Once per Trial for TSInf

        train_buffer, val_buffer = replay_buffer.train_val_split(val_ratio=val_ratio, shuffle=shuffle)
        normalizer = get_normalizer_for_replay_buffer(train_buffer, device)
        processor = SimpleBatchProcessor(device, normalizer=normalizer)
        dynamics_model.normalizer = normalizer

        def eval_func(actions):
            return torch.tensor(dynamics_model_evaluation(state, dynamics_model, actions, num_particles))

        cem_opt = CEMOptimizer(num_samples, elite_size, horizon, num_iterations, lower_bound, upper_bound, alpha,
                               eval_func)

        train_on_replay_buffer(ensemble, train_buffer, val_buffer, optimizer, processor, train_batch_size, train_epochs,
                               log_frequency=train_log_frequency, trial_id=trial_idx)
        trial_length = 0
        trial_return = 0.0

        while True:
            logger.info(f"Starting CEM Optimization for {num_iterations} iterations.")
            start = timer()
            solution = cem_opt.optimize(solution)
            logger.info(f"Took {timer() - start:.3g} seconds.")
            action = solution.detach().cpu().numpy().squeeze()[0]

            logger.info(f"Step {trial_length}# : Taking action {action}")

            action = np.clip(action, lower_bound_np, upper_bound_np)
            next_state, reward, done, info = env.step(action)
            replay_buffer.add(state, action, next_state, reward, done)

            trial_length += 1

            if not done:
                trial_return = reward + trial_gamma * trial_return
            else:
                trial_lengths.append(trial_length)
                trial_returns.append(trial_return)
                logger.info(f"Trial {trial_idx} over after {trial_length} steps with total return {trial_return}.")

                if trial_return > best_trial_return:
                    logger.info("New best trial.")
                    best_trial_return = trial_return
                    best_trial_length = trial_length
                    best_trial_id = trial_idx
                break

    logger.info(f"{num_trials} trials done.")
    logger.info(f"Best trial {best_trial_id} with return {best_trial_return} and length {best_trial_length}.")


if __name__ == "__main__":
    # main()
    # random_agent_evaluation()
    run_pets()
