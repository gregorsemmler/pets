# Probabilistic Ensemble  Trajectory Sampling

An implementation of the PETS [1] algorithm in PyTorch. It is a model-based reinforcement learning method to solve control problems in a sample-efficient manner using probabilistic neural network ensembles and the cross-entropy method (CEM) [2] as a gradient-free optimization method. 

The basic structure of the project is as follows:

* `envs` contains an example environment with a continuous implementation of the cartpole environment.
* `common.py` contains various commonly used functions.
* `data.py` contains the logic for the datasets and normalizers.
* `model.py` contains model definitions.
* `optimizer.py` contains an implementation of the CEM optimizer.
* `pets.py` contains the main pets algorithm code.

## References

[[1]](https://proceedings.neurips.cc/paper/2018/file/3de568f8597b94bda53149c7d7f5958c-Paper.pdf) K. Chua, et al. *Deep reinforcement learning in a handful of trials using probabilistic dynamics models*. Advances in neural information processing systems, 2018

[[2]](https://arxiv.org/abs/2008.06389) C. Pinneri, et al. *Sample-efficient cross-entropy method for real-time planning*. arXiv preprint arXiv:2008.06389, 2020.
