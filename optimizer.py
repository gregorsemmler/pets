import torch


class CEMOptimizer(object):
    """
    Implements the Cross-Entropy Method (CEM) [1] algorithm, to be more precise, the CEM_PETS algorithm
    as implemented in [3] and also described in [2].

    [1] R. Rubinstein and W. Davidson. The cross-entropy method for combinatorial and continuous optimization. Methodology and Computing in Applied Probability, 1999.
    [2] C. Pinneri, et al. Sample-efficient cross-entropy method for real-time planning. arXiv preprint arXiv:2008.06389, 2020.
    [3] K. Chua, et al. Deep reinforcement learning in a handful of trials using probabilistic dynamics models. Advances in neural information processing systems, 2018
    """

    def __init__(self, num_samples, elite_size, horizon, num_iterations, lower_bound: torch.tensor,
                 upper_bound: torch.tensor, alpha, cost_function):
        self.num_samples = num_samples
        self.elite_size = elite_size
        self.horizon = horizon
        self.num_iterations = num_iterations
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound
        self.alpha = alpha
        self.cost_function = cost_function

    def optimize(self, initial_mean=None):
        # https://github.com/kchua/handful-of-trials/blob/77fd8802cc30b7683f0227c90527b5414c0df34c/dmbrl/controllers/MPC.py#L131-L132
        mean = initial_mean if initial_mean is not None else (self.lower_bound + self.upper_bound) / 2
        var = (self.upper_bound - self.lower_bound) ** 2 / 16
        best_solution = torch.empty_like(mean)
        best_value = float("-inf")

        # https://github.com/kchua/handful-of-trials/blob/77fd8802cc30b7683f0227c90527b5414c0df34c/dmbrl/misc/optimizers/cem.py#L122
        lb_dist, ub_dist = mean - self.lower_bound, self.upper_bound - mean
        constrained_var = torch.min(torch.min(torch.square(lb_dist / 2), torch.square(ub_dist / 2)), var)

        new_mean = None
        for _ in range(self.num_iterations):
            sampled = torch.zeros((self.num_samples,) + mean.shape)
            sampled = torch.nn.init.trunc_normal_(sampled) * torch.sqrt(constrained_var) + mean

            costs = self.cost_function(sampled)
            best_values, best_indices = costs.topk(self.elite_size)
            elite = sampled[best_indices]

            new_mean = torch.mean(elite, dim=0)
            new_var = torch.var(elite, dim=0)

            mean = self.alpha * mean + (1 - self.alpha) * new_mean
            var = self.alpha * var + (1 - self.alpha) * new_var

            best_iter_val = best_values[0]
            if best_iter_val > best_value:
                best_value = best_iter_val

        return new_mean
