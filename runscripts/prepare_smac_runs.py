import ast
import os
import pandas as pd
import yaml


SMAC_RESULTS = "results/smac"
INCUMBENTS = "runscripts/configs/smac_incumbents"


if __name__ == "__main__":
    for experiment in os.listdir(SMAC_RESULTS):
        directory = os.path.join(SMAC_RESULTS, experiment)
        if not os.path.isdir(directory):
            continue

        incumbent_path = os.path.join(directory, "0", "42", "incumbent.json")

        if not os.path.isfile(incumbent_path):
            continue

        with open(incumbent_path, "r") as incumbent_file:
            incumbent = incumbent_file.readlines()[-1]

            incumbent = incumbent.replace("true", "True").replace("false", "False")

        incumbent_dict = ast.literal_eval(incumbent)
        incumbent_dict = incumbent_dict["config"] 

        incumbent_yaml = yaml.dump({"hp_config": incumbent_dict}, default_flow_style=False)

        config_path = os.path.join(INCUMBENTS, f"{experiment}.yaml")
        with open(config_path, "w") as config_file:
            config_file.write(incumbent_yaml)

