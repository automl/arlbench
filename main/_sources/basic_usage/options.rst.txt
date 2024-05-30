Options for ARLBench 
====================

A given training run in ARLBench can be configured on two levels: the lower one is the configuration that happens via the AutoRL tool we benchmark while the upper level decides the setting we test the AutoRL tool in.
The high level configuration takes place via the 'autorl' keys in the configuration file. These are the available options:

- **seed**: The seed for the random number generator 
- **env_framework**: Environment framework to use. Currently supported: gymnax, envpool, brax, xland
- **env_name**: The name of the environment to use
- **env_kwargs**: Additional keyword arguments for the environment
- **eval_env_kwargs**: Additional keyword arguments for the evaluation environment
- **n_envs**: Number of environments to use in parallel
- **algorithm**: The algorithm to use. Currently supported: dqn, ppo, sac
- **cnn_policy**: Whether to use a CNN policy
- **deterministic_eval**: Whether to use deterministic evaluation. This diables exploration behaviors in evaluation.
- **nas_config**: Configuration for architecture
- **checkpoint**: A list of elements the checkpoint should contain 
- **checkpoint_name**: The name of the checkpoint
- **checkpoint_dir**: The directory to save the checkpoint in
- **objectives**: The objectives to optimize for. Currently supported: reward_mean, reward_std, runtime, emissions
- **optimize_objectives**: Whether to maximize or minimize the objectives
- **state_features**: The features of the RL algorithm's state to return
- **n_steps**: The number of steps in the configuration schedule. Using 1 will result in a static configuration
- **n_total_timesteps**: The total number of timesteps to train in each schedule interval
- **n_eval_steps**: The number of steps to evaluate the agent for
- **n_eval_episodes**: The number of episodes to evaluate the agent for

The low level configuration options can be found in the 'hp_config' key set, containing the configurable hyperparameters and architecture of each algorithm. Please refer to the search space overview for more information.
