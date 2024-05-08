from arlbench.utils import config_space_to_yaml, save_defaults_to_yaml
from arlbench.core.algorithms import DQN, PPO, SAC


def create_algorithm_configs(seed: int = 0):
    for algorithm in [DQN, PPO, SAC]:
        hp_search_space = algorithm.get_hpo_search_space()
        hp_config_space = algorithm.get_hpo_config_space()
        nas_config_space = algorithm.get_nas_config_space()
        search_space = config_space_to_yaml(hp_search_space, config_key="hp_config", seed=seed)
        with open(
            f"./runscripts/configs/search_space/{algorithm.name}.yaml", "w"
        ) as yaml_file:
            yaml_file.write(search_space)

        default_config = save_defaults_to_yaml(
            hp_config_space, nas_config_space, algorithm.name
        )
        default_config = "# @package _global_\n" + default_config

        with open(
            f"./runscripts/configs/algorithm/{algorithm.name}.yaml", "w"
        ) as yaml_file:
            yaml_file.write(default_config)


if __name__ == "__main__":
    create_algorithm_configs(seed=0)
