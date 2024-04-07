from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from arlbench.core.environments import AutoRLEnv


class AutoRLWrapper:
    """Base class for AutoRL wrappers."""

    def __init__(self, env: AutoRLEnv):
        self._env = env

    # provide proxy access to regular attributes of wrapped object
    def __getattr__(self, name):
        return getattr(self._env, name)
