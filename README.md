<p align="center">
    <a href="./docs/images/logo_lm.png#gh-light-mode-only">
        <img src="./docs/images/logo_lm.png#gh-light-mode-only" alt="ARLBench Logo" width="80%"/>
    </a>
    <a href="./docs/images/logo_dm.png#gh-dark-mode-only">
        <img src="./docs/images/logo_dm.png#gh-dark-mode-only" alt="ARLBench Logo" width="80%"/>
    </a>
</p>

<div align="center">
    
<!--- [![PyPI Version](https://img.shields.io/pypi/v/arlbench.svg)](https://pypi.python.org/pypi/arlbench) -->
[![Test](https://github.com/automl/arlbench/actions/workflows/pytest.yaml/badge.svg)](https://github.com/automl/arlbench/actions/workflows/pytest.yaml)
[![Doc Status](https://github.com/automl/arlbench/actions/workflows/docs.yaml/badge.svg)](https://github.com/automl/arlbench/actions/workflows/docs.yaml)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![Commitizen friendly](https://img.shields.io/badge/commitizen-friendly-brightgreen.svg)](http://commitizen.github.io/cz-cli/)
    
</div>

<div align="center">
    <h3>
      <a href="#features">Features</a> |
      <a href="#installation">Installation</a> |
      <a href="#quickstart">Quickstart</a> |
      <a href="#cite-us">Cite Us</a>
    </h3>
</div>

---

# 🦾 Automated Reinforcement Learning Benchmark

The ARLBench is a benchmark for HPO in RL - evaluate your HPO methods fast and on a representative number of environments! For more information, see our [documentation](https://automl.github.io/arlbench/main/).

## Features

- **Lightning-fast JAX-Based implementations of DQN, PPO, and SAC**
- **Compatible with many different environment domains via Gymnax, XLand and EnvPool**
- **Representative benchmark set of HPO settings** 

## Installation

There are currently two different ways to install ARLBench. 
Whichever you choose, we recommend to create a virtual environment for the installation:

```bash
conda create -n arlbench python=3.10
conda activate arlbench
```

<details>
<summary>PyPI</summary>
You can install ARLBench using `pip`:

```bash
pip install arlbench
```

If you want to use envpool environments (not currently supported for Mac!), instead choose:
```bash
pip install arlbench[envpool]
```

</details>

<details>
<summary>From source: GitHub</summary>
First, you need to clone the ARLBench reopsitory:

```bash
git clone git@github.com:automl/arlbench.git
cd arlbench
```

Then you can install the benchmark. For the base version, use:
```bash
make install
```
For the envpool functionality (not available on Mac!), instead use:

```bash
make install-envpool
```
</details>

> [!CAUTION]
> Windows is currently not supported and also not tested. We recommend using the Linux subsytem if you're on a Windows machine.

If you want to run the ARLBench on GPU, we recommend you check out the [JAX installation guide](https://jax.readthedocs.io/en/latest/installation.html) to see how you can install the correct version for your GPU setup.

## Quickstart

Here are the two ways you can use ARLBench: via the command line or as an environment. To see them in action, take a look at our [examples](https://github.com/automl/arlbench/tree/main/examples).

### Use the CLI

We provide a command line script for black-box configuration in ARLBench. Simply run:
```bash
python run_arlbench.py
```

You can use the hydra command line syntax to override some of the configuration like this:
```bash
python run_arlbench.py algorithm=ppo
```

Or run multiple different versions after one another:
```bash
python run_arlbench.py -m autorl.seed=0,1,2,3,4
```

We recommend you create your own custom config files if using the CLI. Our [examples](https://github.com/automl/arlbench/tree/main/examples) can show you how these can look.

### Use the AutoRL environment

If you want to have specific control over the ARLBench loop, want to do dynamic configuration or learn based on the agent state, you should use the environment-like interface of ARLBench in your script.

To do so, import ARLBench and use the `AutoRLEnv` to run an RL agent:

```python
from arlbench import AutoRLEnv

env = AutoRLEnv()

obs, info = env.reset()

action = env.config_space.sample_configuration()
obs, objectives, term, trunc, info = env.step(action)
```

Just like with RL agents, you can call 'step' multiple times until termination (which you define via the AutoRLEnv's config). For all configuration options, check out our [documentation](https://automl.github.io/arlbench/main/).

## Cite Us

If you use ARLBench in your work, please cite us:

```bibtex
@misc{beckdierkes24,
  author    = {J. Becktepe and J. Dierkes and C. Benjamins and D. Salinas and A. Mohan and R. Rajan and T. Eimer and F. Hutter and H. Hoos and M. Lindauer},
  title     = {ARLBench},
  year      = {2024},
  url = {https://github.com/automl/arlbench},
```
