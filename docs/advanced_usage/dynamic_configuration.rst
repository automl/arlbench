Dynamic Configuration in ARLBench
==================================

In addition to static approaches, which run the whole training given a fixed configuration, ARLBench supports dynamic configuration approaches. 
These methods, in contrast, can adapt the current hyperparameter configuration during training.
To do this, you can use the CLI or the AutoRL Environment as shown in our examples.

When using the CLI, you have to pass a checkpoint path for the current training state. Then, the training is proceeded using the given configuration.

For the AutoRL Environment, you can set `n_steps` to the number of configuration updates you want to perform during training.
By adjusting the number of training steps (`n_total_timesteps`) accordingly and calling the `step()` function multiple times to perform dynamic configuration.
