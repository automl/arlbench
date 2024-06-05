"""Top-level package for ARLBench."""

import datetime

from .arlbench import run_arlbench
from .autorl.autorl_env import AutoRLEnv

__author__ = """AutoML Hannover"""
__email__ = """automl@ai.uni-hannover.de"""
__version__ = """0.1.1"""
__copyright__ = f"Copyright {datetime.date.today().strftime('%Y')}, AutoML Hannover"

__all__ = ["AutoRLEnv", "run_arlbench"]
