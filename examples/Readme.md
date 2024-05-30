# ARLBench Examples

We provide three different categories of examples:
1. Examples for black-box hyperparameter tuning using the 'hypersweeper' package. 
2. Running a schedule based on a reward heuristic
3. Running a reactive schedule based on the gradient history

We use 'hydra' as a command line interface for these experiments, you'll find the corresponding configurations (including some variations on the algorithms and environments) in the 'configs' directory.
The "hypersweeper_tuning" and "schedules" notebooks can help you run these examples and inspect their results.

## 1. Black-Box HPO

We use the 'hypersweeper' package to demonstrate how ARLBench can be used for black-box HPO. Since it's hydra-based, we simply set up a script which takes a configuration, runs it and returns the evaluation reward at the end. First, use pip to install the hypersweeper:

```bash
pip install hypersweeper
```

You can try a single run of arlbench first:

```bash
python run_arlbench.py
```

To use random search instead, you need to choose the "random_search" config and use the "--multirun" flag:

```bash
python run_arlbench.py --config-name=random_search --multirun
```

Finally, we can also use the state-of-the-art SMAC optimizer by changing to the "smac" config:

```bash
python run_arlbench.py --config-name=smac -m
```

## 2. Heuristic Schedules

We can also use ARLBench to dynamically change the hyperparameter config. We provide a simple example for this in 'run_heuristic_schedule.py': as soon as the agent improves over a certain reward threshold, we decrease the exploration epsilon in DQN a bit. This is likely not the best approach in practice, so feel free to play around with this idea! To see the result, run:

```bash
python run_heuristic_schedule.py
```

## 3. Reactive Schedules

Lastly, we can also adjust the hyperparameters based on algorithm statistics. In 'run_reactive_schedule.py' we spike the learning rate if we see the gradient norm stagnating. See how it works by running:

```bash
python run_reactive_schedule.py
```