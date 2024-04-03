from typing import NamedTuple


class EnvSpec(NamedTuple):
    framework: str
    name: str


ENVS = {
    "CartPole": EnvSpec("gymnax", "CartPole-v1"),
    "Pendulum": EnvSpec("gymnax", "Pendulum-v1"),
    # TODO add more
}


def make_env(env_name):
    if env_name not in ENVS.keys():
        raise ValueError(f"Invalid environment: {env_name}")
    
    env_spec = ENVS[env_name]

    if env_spec.framework == "gymnax":
        import gymnax

        env = gymnax.make(env_spec.name)

    elif env_spec.framework == "envpool":
        import envpool

        env = envpool.make(env_spec.name)

    else:
        raise ValueError(f"Invalid framework: {env.framework}")
    
