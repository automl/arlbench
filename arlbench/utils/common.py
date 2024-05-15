from __future__ import annotations

import ConfigSpace
import gymnasium
import gymnasium.spaces as gym_spaces
import gymnax.environments.spaces as gymnax_spaces
import numpy as np
import jax.numpy as jnp
import yaml


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


def save_defaults_to_yaml(hp_config_space, nas_config_sapce, algorithm: str):
    yaml_dict = {
        'algorithm': algorithm,
        'hp_config': {},
        'nas_config': {}
    }

    def add_hps(config_space, config_key):
        for hp_name, hp in config_space.items():
            if isinstance(hp, ConfigSpace.UniformIntegerHyperparameter):
                yaml_dict[config_key][hp_name] = int(hp.default_value)
            elif isinstance(hp, ConfigSpace.UniformFloatHyperparameter):
                yaml_dict[config_key][hp_name] = float(hp.default_value)
            elif isinstance(hp, ConfigSpace.CategoricalHyperparameter):
                if isinstance(hp.default_value, bool):
                    yaml_dict[config_key][hp_name] = bool(hp.default_value)
                elif isinstance(hp.default_value, int):
                    yaml_dict[config_key][hp_name] = int(hp.default_value)
                elif isinstance(hp.default_value, float):
                    yaml_dict[config_key][hp_name] = float(hp.default_value)
                else:
                    yaml_dict[config_key][hp_name] = str(hp.default_value)

    add_hps(hp_config_space, "hp_config")
    add_hps(nas_config_sapce, "nas_config")
        
    return yaml.dump(yaml_dict, sort_keys=False)


def config_space_to_yaml(config_space: ConfigSpace.ConfigurationSpace, config_key: str = 'hp_config', seed: int = 0):
    yaml_dict = {
        'seed': seed,
        'hyperparameters': {},
        'conditions': []
    }
    for hp_name, hp in config_space.items():
        if hp_name == "normalize_observations":
            continue

        hp_key = f"{config_key}.{hp_name}"
        if isinstance(hp, ConfigSpace.UniformIntegerHyperparameter):
            yaml_dict['hyperparameters'][hp_key] = {
                'type': 'uniform_int',
                'upper': int(hp.upper),
                'lower': int(hp.lower),
                'default': int(hp.default_value),
                'log': bool(hp.log)
            }
        elif isinstance(hp, ConfigSpace.UniformFloatHyperparameter):
            yaml_dict['hyperparameters'][hp_key] = {
                'type': 'uniform_float',
                'upper': float(hp.upper),
                'lower': float(hp.lower),
                'default': float(hp.default_value),
                'log': bool(hp.log)
            }
        elif isinstance(hp, ConfigSpace.CategoricalHyperparameter):
            try:
                if len(hp.choices) == 2:    # assume bool
                    param = {
                            'type': 'categorical',
                            'choices': [bool(c) for c in hp.choices],
                            'default': bool(hp.default_value)
                        }
                else:
                    param = {
                        'type': 'categorical',
                        'choices': [int(c) for c in hp.choices],
                        'default': int(hp.default_value)
                    }
            except:
                param = {
                    'type': 'categorical',
                    'choices': [str(c) for c in hp.choices],
                    'default': str(hp.default_value)
                }
            yaml_dict['hyperparameters'][hp_key] = param

    # This part is experimental
    for c in config_space.get_conditions():
        cond = {
            "child": f"{config_key}.{str(c.child.name)}",
            "parent": f"{config_key}.{str(c.parent.name)}",
            "value": bool(c.value)
        }
        if isinstance(c, ConfigSpace.EqualsCondition):
            cond["type"] = "EQ"
        else:
            raise ValueError("Only EqualsCondition is supported.")
        
        yaml_dict['conditions'].append(cond)

    return yaml.dump(yaml_dict, sort_keys=False, default_flow_style=False)


def config_space_to_gymnasium_space(
    config_space: ConfigSpace.ConfigurationSpace, seed=None
) -> gymnasium.spaces.Dict:
    spaces = {}

    for hp_name, hp in config_space._hyperparameters.items():
        if isinstance(hp, ConfigSpace.UniformFloatHyperparameter):
            spaces[hp_name] = gymnasium.spaces.Box(
                low=hp.lower, high=hp.upper, seed=seed, dtype=np.float32
            )
        elif isinstance(hp, ConfigSpace.UniformIntegerHyperparameter):
            spaces[hp_name] = gymnasium.spaces.Box(
                low=hp.lower, high=hp.upper, seed=seed, dtype=np.int32
            )
        elif isinstance(hp, ConfigSpace.CategoricalHyperparameter):
            spaces[hp_name] = gymnasium.spaces.Discrete(
                n=hp.num_choices, start=0, seed=seed
            )
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
        return gymnax_spaces.Dict(
            {k: gymnasium_space_to_gymnax_space(v) for k, v in space.spaces}
        )
    if isinstance(space, gym_spaces.Tuple):
        return gymnax_spaces.Tuple(space.spaces)

    raise NotImplementedError(f"Conversion of {space.__class__.__name__} not supported")


def flatten_dict(d):
    """Flatten a nested dictionary into a tuple containing all items."""
    values = []
    for value in d.values():
        if isinstance(value, dict):
            values.extend(flatten_dict(value))
        else:
            values.append(value)
    return tuple(values)


def recursive_concat(dict1: dict, dict2: dict, axis: int = 0):
    concat_dict = {}

    assert dict1.keys() == dict2.keys(), "Dictionaries have different sets of keys"

    for key in dict1:
        if isinstance(dict1[key], dict) and isinstance(dict2[key], dict):
            concat_dict[key] = recursive_concat(dict1[key], dict2[key], axis)
        else:
            concat_dict[key] = jnp.concatenate([dict1[key], dict2[key]], axis=axis)

    return concat_dict


def tuple_concat(tuple1: tuple, tuple2: tuple, axis: int = 0):
    assert len(tuple1) == len(tuple2), "Tuples must be of the same length"

    concatenated_tuple = tuple(
        {key: jnp.concatenate([d1[key], d2[key]], axis=axis) for key in d1}
        for d1, d2 in zip(tuple1, tuple2)
    )

    return concatenated_tuple
