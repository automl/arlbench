from __future__ import annotations

import ConfigSpace
import gymnasium
import gymnasium.spaces as gym_spaces
import gymnax.environments.spaces as gymnax_spaces
import numpy as np


def to_gymnasium_space(space):
    import gym as old_gym

    if isinstance(space, old_gym.spaces.Box):
        new_space = gymnasium.spaces.Box(
            low=space.low, high=space.high, dtype=space.low.dtype
        )
    elif isinstance(space, old_gym.spaces.Discrete):
        new_space = gymnasium.spaces.Discrete(space.n)
    else:
        raise NotImplementedError
    return new_space


def config_space_to_gymnasium_space(config_space: ConfigSpace.ConfigurationSpace, seed=None) -> gymnasium.spaces.Dict:
    spaces = {}

    for hp_name, hp in config_space._hyperparameters.items():
        if isinstance(hp, ConfigSpace.UniformFloatHyperparameter):
            spaces[hp_name] = gymnasium.spaces.Box(low=hp.lower, high=hp.upper, seed=seed, dtype=np.float32)
        elif isinstance(hp, ConfigSpace.UniformIntegerHyperparameter):
            spaces[hp_name] = gymnasium.spaces.Box(low=hp.lower, high=hp.upper, seed=seed, dtype=np.int32)
        elif isinstance(hp, ConfigSpace.CategoricalHyperparameter):
            spaces[hp_name] = gymnasium.spaces.Discrete(n=hp.num_choices,start=0, seed=seed)
        else:
            raise ValueError(f"Invalid Hyperparameter type for {hp_name}: f{type(hp)}")

    return gymnasium.spaces.Dict(spaces, seed=seed)


def gymnasium_space_to_gymnax_space(space: gym_spaces.Space) -> gymnax_spaces.Space:
    """Convert Gym space to equivalent Gymnax space."""
    if isinstance(space, gym_spaces.Discrete):
        return gymnax_spaces.Discrete(int(space.n))
    if isinstance(space, gym_spaces.Box):
        low = (
            float(space.low)
            if (np.isscalar(space.low) or space.low.size == 1)
            else np.array(space.low)
        )
        high = (
            float(space.high)
            if (np.isscalar(space.high) or space.low.size == 1)
            else np.array(space.high)
        )
        return gymnax_spaces.Box(low, high, space.shape, space.dtype)
    if isinstance(space, gym_spaces.Dict):
        return gymnax_spaces.Dict({
            k: gymnasium_space_to_gymnax_space(v) for k, v in space.spaces
        })
    if isinstance(space, gym_spaces.Tuple):
        return gymnax_spaces.Tuple(space.spaces)

    raise NotImplementedError(
        f"Conversion of {space.__class__.__name__} not supported"
    )

def flatten_dict(d):
    """Flatten a nested dictionary into a tuple containing all items."""
    values = []
    for value in d.values():
        if isinstance(value, dict):
            values.extend(flatten_dict(value))
        else:
            values.append(value)
    return tuple(values)
