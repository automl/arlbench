from typing import NamedTuple
from arlbench.utils.unified_wrappers import GymnaxWrapper, EnvpoolWrapper, Environment


class EnvSpec(NamedTuple):
    framework: str
    name: str


ENVS = {
    "CartPoleGN": EnvSpec("gymnax", "CartPole-v1"),
    "CartPoleEP": EnvSpec("envpool", "CartPole-v1"),
    "PendulumGN": EnvSpec("gymnax", "Pendulum-v1"),
    "PendulumEP": EnvSpec("envpool", "Pendulum-v1"),
    # TODO add more
}


def make_env(env_name, n_envs, seed) -> Environment:
    if env_name not in ENVS.keys():
        raise ValueError(f"Invalid environment: {env_name}")
    
    env_spec = ENVS[env_name]

    if env_spec.framework == "gymnax":
        import gymnax

        env, env_params = gymnax.make(env_spec.name)
        env = GymnaxWrapper(env, n_envs, env_params)
    elif env_spec.framework == "envpool":
        import envpool

        env = envpool.make(env_spec.name, env_type="gymnasium", num_envs=n_envs, seed=seed)
        env = EnvpoolWrapper(env, n_envs)
    else:
        raise ValueError(f"Invalid framework: {env.framework}")
    
    return env
    
