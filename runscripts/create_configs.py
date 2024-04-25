from arlbench.utils import config_space_to_yaml, save_defaults_to_yaml
from arlbench.core.algorithms import DQN, PPO, SAC


def create_configs():
    for algorithm in [DQN, PPO, SAC]:
        config_space = algorithm.get_hpo_config_space()
        search_space = config_space_to_yaml(config_space, config_key="hp_config")
        search_space = "# @package _global_\n" + search_space
        with open(f"./runscripts/configs/search_space/{algorithm.name}.yaml", 'w') as yaml_file:
            yaml_file.write(search_space)

        default_config = save_defaults_to_yaml(config_space, algorithm.name, config_key="hp_config")
        default_config = "# @package _global_\n" + default_config
        with open(f"./runscripts/configs/algorithm/{algorithm.name}.yaml", 'w') as yaml_file:
            yaml_file.write(default_config)


if __name__ == "__main__":
    create_configs()