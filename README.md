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
      <a href="#overview-">Overview</a> |
      <a href="#setup-">Setup</a> |
      <a href="#quickstart-">Quick Start</a> |
      <a href="#examples-">Examples</a> |
      <a href="#citing">Citing</a>
    </h3>
</div>

---

# 🦾 Automated Reinforcement Learning Benchmark

## Overview

## Features

**JAX-Based implementations of DQN, PPO, and SAC**

## Setup

1. **Installation**: You can install ARLBench using `pip`:

```bash
pip install arlbench
```

TODO insert instructions to use SMAC/PBT2 etc

2. **Usage**:

## Quickstart

### Use the CLI

### Use the AutoRL environment

Import ARLBench and use the `AutoRLEnv` to run an RL agent:

```python
from arlbench import AutoRLEnv

env = AutoRLEnv()

obs, info = env.reset()

action = env.config_space.sample_configuration()
obs, objectives, term, trunc, info = env.step(action)
```

## Examples

## Citing
