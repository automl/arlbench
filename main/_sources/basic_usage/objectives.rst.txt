Objectives & Features in ARLBench
======================================

ARLBench allows to configure the objectives you'd like to use for your AutoRL methods. 
These are selected as a list of keywords in the configuration of the AutoRL Environment, e.g. like this:

.. code-block:: bash

    python arlbench.py autorl.objectives=["reward_mean", "discounted_train_reward_mean_gamma_0.9"]

The following objectives are available at the moment:

- **reward_mean**: the mean evaluation reward across a number of evaluation episodes
- **discounted_reward_mean**: the discounted mean evaluation reward across a number of evaluation episodes. The default gamma here is 0.99, but you can specify your own by appending "_gamma_<value>" to the objective name (e.g. discounted_reward_mean_gamma_0.8)
- **reward_std**: the standard deviation of the evaluation rewards across a number of evaluation episodes
- **train_reward_mean**: the mean training reward across a number of training episodes
- **discounted_train_reward_mean**: the discounted mean training reward across a number of training episodes. The default gamma here is 0.99, but you can specify your own by appending "_gamma_<value>" to the objective name (e.g. discounted_train_reward_mean_gamma_0.8)
- **train_reward_std**: the standard deviation of the training rewards across a number of training
- **runtime**: the runtime of the training process
- **emissions**: the CO2 emissions of the training process, tracked using `CodeCarbon <https://github.com/mlco2/codecarbon>`_.

Features work similarly and are intended to be used as additional information about the training run.
You can select them via the 'state_features' key in the configuration of the AutoRL Environment, e.g. like this:

.. code-block:: bash

    python arlbench.py autorl.state_features=["loss_info", "grad_info"]

The following features are available at the moment:
- **grad_info**: information about the gradients during training, i.e. their norm and variance
- **loss_info**: information about the loss during training, i.e. its mean and standard deviation
- **weight_info**: information about the weights of the neural networks used in the RL algorithm, i.e. the norm and variance of the weights and biases in each network
- **prediction_info**: information about the predictions of the neural networks used in the RL algorithm, i.e. the mean and standard deviation of the outputs of each network (like Q-values or log-probs)