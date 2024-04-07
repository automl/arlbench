from __future__ import annotations

import os

from .utils import HandleTermination


def cool_things(cfg):
    """As the name suggests, this does cool things."""
    with HandleTermination(os.getcwd()):
        print(f"Your current config is: {cfg}")


