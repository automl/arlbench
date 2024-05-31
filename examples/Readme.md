# ARLBench Examples

We provide three different categories of examples:
1. Examples for black-box hyperparameter tuning using the 'hypersweeper' package. 
2. Running a schedule based on a reward heuristic
3. Running a reactive schedule based on the gradient history

We use 'hydra' as a command line interface for these experiments, you'll find the corresponding configurations (including some variations on the algorithms and environments) in the 'configs' directory.
The "hypersweeper_tuning" and "schedules" notebooks can help you run these examples and inspect their results in an interactive way.

## 1. Black-Box HPO

We use the [Hypersweeper](https://github.com/automl/hypersweeper/) package to demonstrate how ARLBench can be used for black-box HPO. Since it's hydra-based, we simply set up a script which takes a configuration, runs it and returns the evaluation reward at the end. First, use pip to install the hypersweeper:

```bash
pip install hypersweeper
```

You can try a single run of arlbench first using DQN on CartPole:

```bash
python run_arlbench.py
```

To use random search instead, you should install hypersweeper, choose the "random_search" config and use the "--multirun" flag which is a signal to hydra to engage the sweeper:

```bash
pip install hypersweeper
python run_arlbench.py --config-name=random_search --multirun
```

Finally, we can also use the state-of-the-art SMAC optimizer by changing to the "smac" config:

```bash
python run_arlbench.py --config-name=smac -m
```

You can switch between the environments and algorithms in ARLBench by specifying it in the command line like this:

```bash
python run_arlbench.py --config-name=smac -m environment=cc_cartpole algorithm=ppo search_space=ppo_cc
```

You can see what exactly this command changes by looking into the example configs. In 'configs/algorithm/ppo.yaml', for example, we see the following:

```yaml
# @package _global_
algorithm: ppo
hp_config:
  clip_eps: 0.2
  ent_coef: 0.0
  gae_lambda: 0.95
  gamma: 0.99
  learning_rate: 0.0003
  max_grad_norm: 0.5
  minibatch_size: 64
  n_steps: 128
  normalize_advantage: true
  normalize_observations: false
  update_epochs: 10
  vf_clip_eps: 0.2
  vf_coef: 0.5
nas_config:
  activation: tanh
  hidden_size: 64
```

These are the default arguments for the PPO algorithm in ARLBench. You can also override each of these individually, if you for example want to try a different value for gamma:

```bash
python run_arlbench.py --config-name=smac -m environment=cc_cartpole algorithm=ppo search_space=ppo_cc hp_config.gamma=0.8
```

The search space specification works very similarly via a yaml file, 'configs/search_space/cc_ppo.yaml' contains:
```yaml
seed: 0
hyperparameters:
  hp_config.learning_rate:
    type: uniform_float
    upper: 0.1
    lower: 1.0e-06
    log: true
  hp_config.ent_coef:
    type: uniform_float
    upper: 0.5 
    lower: 0.0
    log: false
  hp_config.minibatch_size:
    type: categorical
    choices: [128, 256, 512]
  hp_config.gae_lambda:
    type: uniform_float
    upper: 0.9999
    lower: 0.8
    log: false
  hp_config.clip_eps:
    type: uniform_float
    upper: 0.5
    lower: 0.0
    log: false
  hp_config.vf_clip_eps:
    type: uniform_float
    upper: 0.5
    lower: 0.0
    log: false
  hp_config.normalize_advantage:
    type: categorical
    choices: [True, False]
  hp_config.vf_coef:
    type: uniform_float
    upper: 1.0
    lower: 0.0
    default: 0.5
    log: false
  hp_config.max_grad_norm:
    type: uniform_float
    upper: 1.0
    lower: 0.0
    log: false
```

This config sets a seed for the search space as well as lists the hyperparameters to configure with the values they can take. This way we can configure the full HPO setting with only yaml files to make the process easy to follow and simple to document for others.


## 2. Heuristic Schedules

We can also use ARLBench to dynamically change the hyperparameter config. We provide a simple example for this in 'run_heuristic_schedule.py': as soon as the agent improves over a certain reward threshold, we decrease the exploration epsilon in DQN a bit. This is likely not the best approach in practice, so feel free to play around with this idea! To see the result, run:

```bash
python run_heuristic_schedule.py
```

Since we now run ARLBench dynamically, we have to think of another configuration option: the settings for dynamic execution. This is configured using the 'autorl' set of keys, our general settings for ARLBench. The default version looks like this:

```yaml
autorl:
  seed: 42
  env_framework: ${environment.framework}
  env_name: ${environment.name}
  env_kwargs: ${environment.kwargs}
  eval_env_kwargs: ${environment.eval_kwargs}
  n_envs: ${environment.n_envs}
  algorithm: ${algorithm}
  cnn_policy: ${environment.cnn_policy}
  nas_config: ${nas_config}
  n_total_timesteps: ${environment.n_total_timesteps}
  checkpoint: []
  checkpoint_name: "default_checkpoint"
  checkpoint_dir: "/tmp"
  state_features: []
  objectives: ["reward_mean"]
  optimize_objectives: "upper"
  n_steps: 1
  n_eval_steps: 100
  n_eval_episodes: 10
```

As you can see, most of the defaults are decided by the environment and algorithm we choose. For dynamic execution, we are interested in the 'n_steps' and 'n_total_timesteps' keys. 
'n_steps' decides how many steps should be taken in the AutoRL Environment - in other words, how many schedule intervals we'd like to have. The 'n_total_timesteps' key then decides the length of each interval.
In the current config, we do a single training interval consisting of the total number of environment steps suggested for our target domain. If we want to instead run a schedule of length 10 with each schedule segment taking 10e4 steps, we can change the configuration like this:

```bash
python run_heuristic_schedule.py autorl.n_steps=10 autorl.n_total_timesteps=10000
```

## 3. Reactive Schedules

Lastly, we can also adjust the hyperparameters based on algorithm statistics. In 'run_reactive_schedule.py' we spike the learning rate if we see the gradient norm stagnating. See how it works by running:

```bash
python run_reactive_schedule.py
```

To actually configure to what information ARLBench returns about the RL algorithm's internal state, we can use the 'state' features key - in this case, we want to add the gradient norm and variance like this:

```bash
python run_reactive_schedule.py "autorl.state_features=['grad_info']"
```

Now we can build a schedule that takes the gradient information into account.