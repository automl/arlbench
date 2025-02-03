from __future__ import annotations

import gymnasium
import numpy as np
from arlbench.utils import config_space_to_gymnasium_space
from ConfigSpace import Categorical, ConfigurationSpace, Float, Integer


def test_config_space_to_gymnasium_space():
    config_space = ConfigurationSpace(
        name="PPOConfigSpace",
        seed=42,
        space={
            "intHP": Integer("intHP", (1, 10), default=5),
            "floatHP": Float("floatHP", (0.0, 1.0), default=0.5),
            "catHP": Categorical("catHP", ["a", "b", "c"], default="a"),
        },
    )

    gym_space = config_space_to_gymnasium_space(config_space)
    assert isinstance(gym_space, gymnasium.spaces.Dict)

    assert isinstance(gym_space["intHP"], gymnasium.spaces.Box)
    assert np.issubdtype(gym_space["intHP"].dtype, np.integer)
    assert np.isclose(gym_space["intHP"].low, 1, atol=1e-4)
    assert np.isclose(gym_space["intHP"].high, 10, atol=1e-4)

    assert isinstance(gym_space["floatHP"], gymnasium.spaces.Box)
    assert np.issubdtype(gym_space["floatHP"].dtype, np.floating)
    assert np.isclose(gym_space["floatHP"].low, 0, atol=1e-4)
    assert np.isclose(gym_space["floatHP"].high, 1, atol=1e-4)

    assert isinstance(gym_space["catHP"], gymnasium.spaces.Discrete)
    assert gym_space["catHP"].n == 3


def test_gym_space_to_gymnax_space():
    # TODO implement
    pass
