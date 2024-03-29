import argparse
import logging
import math
from datetime import datetime
from timeit import default_timer as timer

import gym
import numpy as np
import torch
import torch.nn.functional as F
from gym import envs
from torch.optim import Adam
from torch.utils.tensorboard import SummaryWriter

from common import shift_numpy_array
from data import ReplayBuffer, BatchProcessor, gauss_nll_ensemble_loss, SimpleBatchProcessor, StandardNormalizer, \
    ensemble_loss
from envs.cartpole_continuous import ContinuousCartPoleEnv
from model import DynamicsModel, EnsembleDynamicsModel, EnsembleMode, PolicyEnsemble, MLPEnsemble
from optimizer import CEMOptimizer

logger = logging.getLogger(__name__)


class DummySummaryWriter(object):

    def add_scalar(self, tag, scalar_value, global_step=None, walltime=None):
        pass


PETS_ARG_PARSER = argparse.ArgumentParser()
PETS_ARG_PARSER.add_argument("--device_token", default=None)
PETS_ARG_PARSER.add_argument("--run_id", default=None)
PETS_ARG_PARSER.add_argument("--num_trials", type=int, default=None)
PETS_ARG_PARSER.add_argument("--max_steps", type=int, default=None)
PETS_ARG_PARSER.add_argument("--done_threshold", type=float, default=0.5)
PETS_ARG_PARSER.add_argument("--tensorboardlog", dest="tensorboardlog", action="store_true")
PETS_ARG_PARSER.add_argument("--no_tensorboardlog", dest="tensorboardlog", action="store_false")
PETS_ARG_PARSER.set_defaults(tensorboardlog=False)


def dynamics_model_evaluation(initial_state, dynamics_model: DynamicsModel, horizon_actions, num_particles):
    batch_size, horizon = horizon_actions.shape[:2]
    states = np.tile(initial_state, (batch_size * num_particles,) + tuple([1] * initial_state.ndim))
    average_rewards = np.zeros(batch_size * num_particles, dtype=np.float32)
    done_envs = np.zeros((batch_size * num_particles, ), dtype=bool)
    for time_step in range(horizon):
        action = horizon_actions[:, time_step, :]
        actions = np.repeat(action, num_particles, axis=0)
        with torch.no_grad():
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


class EnsembleTrainer(object):

    def __init__(self, ensemble: PolicyEnsemble, optimizer, writer=None):
        self.ensemble = ensemble
        self.optimizer = optimizer

        self.writer = writer if writer is not None else DummySummaryWriter()
        self.train_batch_idx = 0
        self.val_batch_idx = 0
        self.epoch_idx = 0

    def get_model_out(self, model_out):
        if self.ensemble.predict_rewards:
            means, log_stds, rewards, done_outs = model_out
            nll_out = (means, log_stds)
        else:
            nll_out = model_out
            rewards, done_outs = None, None
        return nll_out, rewards, done_outs

    def process_batches(self, processor, ensemble_batches):
        processed_batches = [processor.process(b) for b in ensemble_batches]
        if processor.process_rewards:
            model_in, target_out, rewards, dones = list(zip(*processed_batches))
        else:
            model_in, target_out = list(zip(*processed_batches))
            rewards, dones = None, None
        return model_in, target_out, rewards, dones

    def fit(self, train_buffer: ReplayBuffer, val_buffer: ReplayBuffer, optimizer, processor: BatchProcessor,
            batch_size, num_epochs, log_frequency=100, trial_id=None, silent=False):

        prev_ensemble_mode = self.ensemble.ensemble_mode
        self.ensemble.ensemble_mode = EnsembleMode.ALL_MEMBERS

        train_batch_losses, val_batch_losses = [], []
        train_epoch_losses, val_epoch_losses = [], []

        trial_id = "" if trial_id is None else f"Trial {trial_id}: "
        # logger.info(
        #     f"{trial_id}Training for {num_epochs} epochs. (TrainSize / ValSize): ({len(train_buffer)}/{len(val_buffer)})")

        for _ in range(num_epochs):

            self.ensemble.train()
            train_epoch_loss = 0.0
            train_batch_count = 0
            if not silent:
                logger.info(f"{trial_id}Training Epoch {self.epoch_idx}.")
            for ensemble_batches in train_buffer.batches(batch_size, self.ensemble.num_members):
                model_in, target_out, rewards, dones = self.process_batches(processor, ensemble_batches)

                model_out = self.ensemble(model_in)

                nll_out, reward_predictions, done_outs = self.get_model_out(model_out)

                nll_loss = gauss_nll_ensemble_loss(nll_out, target_out)
                # https://github.com/kchua/handful-of-trials/blob/77fd8802cc30b7683f0227c90527b5414c0df34c/dmbrl/modeling/models/BNN.py#L182
                log_std_limit_loss = 0.02 * (self.ensemble.max_log_std.sum() - self.ensemble.min_log_std.sum())
                total_loss = nll_loss + log_std_limit_loss

                if rewards is not None:
                    mse_loss = ensemble_loss(F.mse_loss, reward_predictions, rewards)
                    self.writer.add_scalar("train_batch/mse_loss", mse_loss.item(), self.train_batch_idx)
                    total_loss += mse_loss

                if dones is not None:
                    done_loss = ensemble_loss(F.binary_cross_entropy_with_logits, done_outs, dones)
                    self.writer.add_scalar("train_batch/done_loss", done_loss.item(), self.train_batch_idx)
                    total_loss += done_loss

                optimizer.zero_grad()
                total_loss.backward()
                optimizer.step()
                loss_value = total_loss.item()
                self.writer.add_scalar("train_batch/nll_loss", nll_loss.item(), self.train_batch_idx)
                self.writer.add_scalar("train_batch/log_std_limit_loss", log_std_limit_loss.item(), self.train_batch_idx)
                self.writer.add_scalar("train_batch/loss", loss_value, self.train_batch_idx)

                if not silent and self.train_batch_idx % log_frequency == 0:
                    logger.info(
                        f"{trial_id}Train Epoch #{self.epoch_idx} Batch #{self.train_batch_idx} Loss: {loss_value}")

                train_epoch_loss += loss_value
                train_batch_losses.append(loss_value)
                self.train_batch_idx += 1
                train_batch_count += 1

            train_epoch_loss = 0.0 if train_batch_count == 0 else train_epoch_loss / train_batch_count
            train_epoch_losses.append(train_epoch_loss)
            self.writer.add_scalar("train_epoch/loss", train_epoch_loss, self.epoch_idx)

            if not silent:
                logger.info(f"{trial_id}Validation Epoch {self.epoch_idx}.")
            self.ensemble.eval()
            val_epoch_loss = 0.0
            val_batch_count = 0
            for ensemble_batches in val_buffer.batches(batch_size, self.ensemble.num_members):
                model_in, target_out, rewards, dones = self.process_batches(processor, ensemble_batches)

                with torch.no_grad():
                    model_out = self.ensemble(model_in)

                nll_out, reward_predictions, done_outs = self.get_model_out(model_out)

                nll_loss = gauss_nll_ensemble_loss(nll_out, target_out)

                # https://github.com/kchua/handful-of-trials/blob/77fd8802cc30b7683f0227c90527b5414c0df34c/dmbrl/modeling/models/BNN.py#L182
                log_std_limit_loss = 0.02 * (self.ensemble.max_log_std.sum() - self.ensemble.min_log_std.sum())
                total_loss = nll_loss + log_std_limit_loss

                if rewards is not None:
                    mse_loss = ensemble_loss(F.mse_loss, reward_predictions, rewards)
                    self.writer.add_scalar("train_batch/mse_loss", mse_loss.item(), self.train_batch_idx)
                    total_loss += mse_loss

                if dones is not None:
                    done_loss = ensemble_loss(F.binary_cross_entropy_with_logits, done_outs, dones)
                    self.writer.add_scalar("train_batch/done_loss", done_loss.item(), self.train_batch_idx)
                    total_loss += done_loss

                loss_value = total_loss.item()
                self.writer.add_scalar("val_batch/nll_loss", nll_loss.item(), self.train_batch_idx)
                self.writer.add_scalar("val_batch/log_std_limit_loss", log_std_limit_loss.item(), self.train_batch_idx)
                self.writer.add_scalar("val_batch/loss", loss_value, self.val_batch_idx)

                if not silent and self.val_batch_idx % log_frequency == 0:
                    logger.info(f"{trial_id}Val Epoch #{self.epoch_idx} Batch #{self.val_batch_idx} Loss: {loss_value}")

                val_epoch_loss += loss_value
                val_batch_losses.append(loss_value)
                self.val_batch_idx += 1
                val_batch_count += 1

            val_epoch_loss = 0.0 if val_batch_count == 0 else val_epoch_loss / val_batch_count
            val_epoch_losses.append(val_epoch_loss)
            self.writer.add_scalar("val_epoch/loss", val_epoch_loss, self.epoch_idx)

            self.epoch_idx += 1

        self.ensemble.ensemble_mode = prev_ensemble_mode

        return train_batch_losses, val_batch_losses, train_epoch_losses, val_epoch_losses


def run_pets(args):
    logging.basicConfig(level=logging.INFO)

    run_id = args.run_id if args.run_id is not None else f"run_{datetime.now():%d%m%Y_%H%M%S}"
    writer = SummaryWriter(comment=f"-{run_id}") if args.tensorboardlog else DummySummaryWriter()

    env = ContinuousCartPoleEnv()
    state = env.reset()
    state_shape = state.shape
    action_shape = env.action_space.sample().shape

    if len(state_shape) != 1 or len(action_shape) != 1:
        raise ValueError("State and action shape need to have dimension 1")

    state_dim = state_shape[0]
    action_dim = action_shape[0]

    replay_buffer_size = 3000
    replay_buffer = ReplayBuffer(replay_buffer_size, state_shape,  action_shape)

    # Train parameters
    train_batch_size = 32
    # train_batch_size = 128
    train_epochs = 50
    # train_epochs = 1000
    train_lr = 1e-3
    # l2_regularization = 5e-5
    l2_regularization = 0
    val_ratio = 0.05
    shuffle = True
    train_log_frequency = 1

    num_random_steps = 200
    # Fill replay buffer with initial data from random actions
    play_and_add_to_buffer(env, lambda x: env.action_space.sample(), replay_buffer, num_random_steps)

    num_ensemble_members = 5
    activation = "elu"
    fully_params = [200, 200, 200]
    reward_params = [100, 100]
    # reward_params = None
    balance_classes = reward_params is not None
    # ensemble_mode = EnsembleMode.SHUFFLED_MEMBER
    ensemble_mode = EnsembleMode.FIXED_MEMBER
    done_th = args.done_threshold

    if args.device_token is None:
        device_token = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device_token = args.device_token

    device = torch.device(device_token)
    ensemble = MLPEnsemble(state_dim, action_dim, num_ensemble_members, ensemble_mode=ensemble_mode,
                           fully_params=fully_params, activation=activation, reward_params=reward_params).to(device)
    optimizer = Adam(ensemble.parameters(), lr=train_lr, weight_decay=l2_regularization)
    dynamics_model = EnsembleDynamicsModel(ensemble, env, device, done_threshold=done_th)

    # CEM Options
    num_samples = 500
    elite_size = 50
    horizon = 15
    num_iterations = 5
    lower_bound_np, upper_bound_np = env.action_space.low, env.action_space.high
    alpha = 0.1
    num_particles = 20

    lower_bound = torch.tensor(np.tile(lower_bound_np, (horizon, 1)))
    upper_bound = torch.tensor(np.tile(upper_bound_np, (horizon, 1)))

    initial_solution = (lower_bound + upper_bound) / 2

    trial_gamma = 1.0

    trial_lengths = []
    trial_returns = []
    best_trial_return = float("-inf")
    best_trial_length = 0
    best_trial_id = None

    trainer = EnsembleTrainer(ensemble, optimizer, writer)

    num_trials = 5 if args.num_trials is None else args.num_trials
    silent_log = True
    max_steps = math.inf if args.max_steps is None else args.max_steps

    for trial_idx in range(num_trials):
        logger.info(f"Starting trial {trial_idx}.")
        state = env.reset()
        solution = initial_solution.clone()
        if ensemble_mode == EnsembleMode.SHUFFLED_MEMBER:
            ensemble.shuffle_ids(num_samples * num_particles)  # Once per Trial for TSInf

        train_buffer, val_buffer = replay_buffer.train_val_split(val_ratio=val_ratio, shuffle=shuffle,
                                                                 balance_classes=balance_classes)
        normalizer = get_normalizer_for_replay_buffer(train_buffer, device)
        processor = SimpleBatchProcessor(device, normalizer=normalizer, process_rewards=ensemble.predict_rewards)
        dynamics_model.normalizer = normalizer

        def eval_func(actions):
            actions_np = actions.detach().cpu().numpy()
            return torch.tensor(dynamics_model_evaluation(state, dynamics_model, actions_np, num_particles))

        cem_opt = CEMOptimizer(num_samples, elite_size, horizon, num_iterations, lower_bound, upper_bound, alpha,
                               eval_func)

        trainer.fit(train_buffer, val_buffer, optimizer, processor, train_batch_size, train_epochs,
                    log_frequency=train_log_frequency, trial_id=trial_idx, silent=silent_log)
        trial_length = 0
        trial_return = 0.0

        while True:
            if not silent_log:
                logger.info(f"Starting CEM Optimization for {num_iterations} iterations.")
                start = timer()

            solution = cem_opt.optimize(solution)

            if not silent_log:
                logger.info(f"Took {timer() - start:.3g} seconds.")
            solution_np = solution.detach().cpu().numpy()
            action = solution_np.squeeze()[0]

            # Shift solution by one element for next round
            solution_np = shift_numpy_array(solution_np, -1, fill_value=float(initial_solution[0]))
            solution = torch.from_numpy(solution_np)

            logger.info(f"Trial {trial_idx}: Step {trial_length}# : Taking action {action}")

            action = np.clip(action, lower_bound_np, upper_bound_np)
            next_state, reward, done, info = env.step(action)
            replay_buffer.add(state, action, next_state, reward, done)

            state = next_state

            trial_length += 1

            if trial_length >= max_steps:
                logger.info(f"Maximum number of steps ({max_steps}) reached. Ending trial.")
            end_trial = done or trial_length >= max_steps

            if not done:
                trial_return = reward + trial_gamma * trial_return

            if end_trial:
                trial_lengths.append(trial_length)
                trial_returns.append(trial_return)
                logger.info(f"Trial {trial_idx} over after {trial_length} steps with total return {trial_return}.")

                writer.add_scalar("trial_return", trial_return, trial_idx)
                writer.add_scalar("trial_length", trial_length,  trial_idx)

                if trial_return > best_trial_return:
                    logger.info("New best trial.")
                    best_trial_return = trial_return
                    best_trial_length = trial_length
                    best_trial_id = trial_idx
                break

    logger.info(f"{num_trials} trials done.")
    logger.info(f"Best trial {best_trial_id} with return {best_trial_return} and length {best_trial_length}.")


if __name__ == "__main__":
    cmd_args = PETS_ARG_PARSER.parse_args()
    run_pets(cmd_args)
