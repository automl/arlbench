Dynamic Configuration in ARLBench
==================================

In addition to static approaches, which run the whole training given a fixed configuration, ARLBench supports **dynamic configuration approaches**. 
These methods, in contrast, can adapt the current hyperparameter configuration **during training**.
To do this, you can use the CLI or the AutoRL Environment as shown in our examples.

When using the CLI, you have to **pass a checkpoint path** for the current training state. 
Then, the training is proceeded using this training state with a new configuration.
This is especially useful for highly parallelizable dynamic tuning methods, e.g. population based methods.

For the AutoRL Environment, you can set `n_steps` to the **number of configuration updates** you want to perform during training.
You should also adjust (`n_total_timesteps`) accordingly down to 1/`n_steps` in your settings. 
Then calling the `step()` function multiple times until termination will perform the same dynamic configuration as with the CLI.
