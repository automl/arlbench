from arlbench.utils import config_space_to_gymnasium_space
from ConfigSpace import ConfigurationSpace, Integer, Float, Categorical
import gymnasium

def test_config_space_to_gymnasium_space():
    config_space = ConfigurationSpace(
        name="PPOConfigSpace",
        seed=42,
        space={
            "intHP": Integer("intHP", (1, 10), default=5),
            "floatHP": Float("floatHP", (0., 1.), default=0.5),
            "catHP": Categorical("catHP", ["a", "b", "c"], default="a")
        },
    )

    gym_space = config_space_to_gymnasium_space(config_space)
    print(gym_space)
    assert isinstance(gym_space, gymnasium.spaces.Dict)
