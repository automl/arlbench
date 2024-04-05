from arlbench.environments import AutoRLEnv


class AutoRLWrapper(object):
    """Base class for AutoRL wrappers."""

    def __init__(self, env: AutoRLEnv):
        self._env = env

    # provide proxy access to regular attributes of wrapped object
    def __getattr__(self, name):
        return getattr(self._env, name)
