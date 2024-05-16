<p align="center">
    <a href="./docs/images/logo_lm.png#gh-light-mode-only">
        <img src="./docs/images/logo_lm.png#gh-light-mode-only" alt="ARLBench Logo" width="80%"/>
    </a>
    <a href="./docs/images/logo_dm.png#gh-dark-mode-only">
        <img src="./docs/images/logo_dm.png#gh-dark-mode-only" alt="ARLBench Logo" width="80%"/>
    </a>
</p>

<!--- [![PyPI Version](https://img.shields.io/pypi/v/arlbench.svg)](https://pypi.python.org/pypi/arlbench) -->
[![Test](https://github.com/automl/arlbench/actions/workflows/pytest.yaml/badge.svg)](https://github.com/automl/arlbench/actions/workflows/pytest.yaml)
[![Doc Status](https://github.com/automl/arlbench/actions/workflows/docs.yaml/badge.svg)](https://github.com/automl/arlbench/actions/workflows/docs.yaml)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![Commitizen friendly](https://img.shields.io/badge/commitizen-friendly-brightgreen.svg)](http://commitizen.github.io/cz-cli/)

<div align="center">
    <h3>
      <a href="#features-">Features</a> |
      <a href="#setup-">Setup</a> |
      <a href="#quickstart-">Quickstart</a> |
      <a href="#citing">Cite Us</a>
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

There are currently two different ways to install ARLBench:

<details>
<summary>After acceptance: PyPI</summary>
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

We recommend to create a virtual environment for the installation:
```bash
conda create -n arlbench python=3.10
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

## Quickstart

Here are the two ways you can use ARLBench: via the command line or as an environment. To see them in action, take a look at our [examples](https://github.com/automl/arlbench/tree/main/examples).

### Use the CLI

TODO: where does this even exist?

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
