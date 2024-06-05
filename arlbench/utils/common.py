"""Common utility functions for the ARLBench package."""
from __future__ import annotations

import ConfigSpace
import gymnasium
import gymnasium.spaces as gym_spaces
import gymnax.environments.spaces as gymnax_spaces
import jax.numpy as jnp
import numpy as np
import yaml
from ConfigSpace import ConfigurationSpace

CAT_HP_CHOICES = 2


def save_defaults_to_yaml(
        hp_config_space: ConfigurationSpace,
        nas_config_sapce: ConfigurationSpace,
        algorithm: str
    ) -> str:
    """Extracts the default values of the hp_config_space and nas_config_sapce and
    returns a yaml file.

    Args:
        hp_config_space (ConfigurationSpace): The hyperparameter configuration space
        of the algorithm.
        nas_config_sapce (ConfigurationSpace): The neural architecture configuration
        space of the algorithm.
        algorithm (str): The name of the algorithm.

    Returns:
        str: yaml string.
    """
    yaml_dict = {"algorithm": algorithm, "hp_config": {}, "nas_config": {}}

    def add_hps(config_space: ConfigurationSpace, config_key: str) -> None:
        """Adds hyperparameter defaults to a dictionary."""
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


def config_space_to_yaml(
    config_space: ConfigurationSpace,
    config_key: str = "hp_config",
    seed: int = 0,
) -> str:
    """Converts a ConfigSpace object to yaml.

    Args:
        config_space (ConfigurationSpace): Configuration space object.
        config_key (str, optional): Key for the hyperparameters.
        Defaults to "hp_config".
        seed (int, optional): Configuration space seed to write to yaml. Defaults to 0.


    Returns:
        _type_: _description_
    """
    yaml_dict = {"seed": seed, "hyperparameters": {}, "conditions": []}
    for hp_name, hp in config_space.items():
        if hp_name == "normalize_observations":
            continue

        hp_key = f"{config_key}.{hp_name}"
        if isinstance(hp, ConfigSpace.UniformIntegerHyperparameter):
            yaml_dict["hyperparameters"][hp_key] = {
                "type": "uniform_int",
                "upper": int(hp.upper),
                "lower": int(hp.lower),
                "default": int(hp.default_value),
                "log": bool(hp.log),
            }
        elif isinstance(hp, ConfigSpace.UniformFloatHyperparameter):
            yaml_dict["hyperparameters"][hp_key] = {
                "type": "uniform_float",
                "upper": float(hp.upper),
                "lower": float(hp.lower),
                "default": float(hp.default_value),
                "log": bool(hp.log),
            }
        elif isinstance(hp, ConfigSpace.CategoricalHyperparameter):
            try:
                if len(hp.choices) == CAT_HP_CHOICES:  # assume bool
                    param = {
                        "type": "categorical",
                        "choices": [bool(c) for c in hp.choices],
                        "default": bool(hp.default_value),
                    }
                else:
                    param = {
                        "type": "categorical",
                        "choices": [int(c) for c in hp.choices],
                        "default": int(hp.default_value),
                    }
            except TypeError:
                param = {
                    "type": "categorical",
                    "choices": [str(c) for c in hp.choices],
                    "default": str(hp.default_value),
                }
            yaml_dict["hyperparameters"][hp_key] = param

    # This part is experimental
    for c in config_space.get_conditions():
        cond = {
            "child": f"{config_key}.{c.child.name!s}",
            "parent": f"{config_key}.{c.parent.name!s}",
            "value": bool(c.value),
        }
        if isinstance(c, ConfigSpace.EqualsCondition):
            cond["type"] = "EQ"
        else:
            raise ValueError("Only EqualsCondition is supported.")

        yaml_dict["conditions"].append(cond)

    return yaml.dump(yaml_dict, sort_keys=False, default_flow_style=False)


def config_space_to_gymnasium_space(
    config_space: ConfigurationSpace, seed: int | None = None
) -> gymnasium.spaces.Dict:
    """Converts a configuration space to a gymnasium space.

    Args:
        config_space (ConfigurationSpace): Configuration space.
        seed (int | None, optional): Seed for the gymnasium space. Defaults to None.

    Returns:
        gymnasium.spaces.Dict: Gymnasium space.
    """
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
    """Converst a gymnasium space to a gymnax space.

    Args:
        space (Space): Gymnasium space.

    Returns:
        gymnax_spaces.Space: Gymnax space.
    """
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


def recursive_concat(dict1: dict, dict2: dict, axis: int = 0) -> dict:
    """Recursively concatenates two dictionaries value-wise for same keys.

    Args:
        dict1 (dict): First dictionary.
        dict2 (dict): Second dictionary.
        axis (int, optional): Concat axis. Defaults to 0.

    Returns:
        dict: Concatenated dictionary.
    """
    concat_dict = {}

    assert dict1.keys() == dict2.keys(), "Dictionaries have different sets of keys"

    for key in dict1:
        if isinstance(dict1[key], dict) and isinstance(dict2[key], dict):
            concat_dict[key] = recursive_concat(dict1[key], dict2[key], axis)
        else:
            concat_dict[key] = jnp.concatenate([dict1[key], dict2[key]], axis=axis)

    return concat_dict


def tuple_concat(tuple1: tuple, tuple2: tuple, axis: int = 0) -> tuple:
    """Concatenates two tuples element-wise.

    Args:
        tuple1 (tuple): First tuple.
        tuple2 (tuple): Second tuple.
        axis (int, optional): Concat axis. Defaults to 0.

    Returns:
        tuple: Concatenated tuple.
    """
    assert len(tuple1) == len(tuple2), "Tuples must be of the same length"

    return tuple(
        {key: jnp.concatenate([d1[key], d2[key]], axis=axis) for key in d1}
        for d1, d2 in zip(tuple1, tuple2, strict=False)
    )
