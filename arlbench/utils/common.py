import numpy as np
import ConfigSpace
import gymnasium


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