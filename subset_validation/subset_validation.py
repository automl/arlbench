import os
import pandas as pd
import ast
from collections import defaultdict
import seaborn as sns
import matplotlib.pyplot as plt
import random


SEED = 0
OPTIMIZER_RESULTS = {
    "SMAC_BO": "results/smac",
    "SMAC_MultiFidelity": "results/smac_mf",
    "RandomSearch": "results/rs",
    "PBT": "results/pbt"
}

SUBSETS = {
    "ppo": ["MountainCar-v0", "CartPole-v1"],
    "dqn": ["MountainCar-v0", "MiniGrid-EmptyRandom-5x5"],
    "sac": ["humanoid", "halfcheetah"],
}


def read_optimizer_results(optimizer: str) -> pd.DataFrame:
    scores = []
    path = OPTIMIZER_RESULTS[optimizer]

    for result_dir in os.listdir(path):
        if os.path.isfile(os.path.join(path, result_dir)):
            continue
        print(result_dir)
        algorithm = result_dir.split("_")[0]
        environment = result_dir.split("_")[1]

        incumbent_path = os.path.join(path, result_dir, str(SEED), "42", "incumbent.json")
        if not os.path.isfile(incumbent_path):
            continue

        with open(incumbent_path, "r") as incumbent_file:
            incumbent = incumbent_file.readlines()[-1]

            incumbent = incumbent.replace("true", "True").replace("false", "False")

        incumbent_dict = ast.literal_eval(incumbent)
        score = incumbent_dict["score"]

        # During the optimization, we are minimizing the scores,
        # so we have to get the actual reward
        score = -1 * score

        scores += [{
            "algorithm": algorithm,
            "environment": environment,
            "score": score,
            "optimizer": optimizer
        }]
    scores = pd.DataFrame(scores)
    scores.to_csv(os.path.join(path, "combined.csv"), index=False)
    return scores

def read_optimizer_data_per_algorithm():
    results = []
    for optimizer in OPTIMIZER_RESULTS:
        data = read_optimizer_results(optimizer)
        data["optimizer"] = optimizer
        results += [data]

    results = pd.concat(results)

    algorithm_results = {}
    for algorithm in SUBSETS:
        algorithm_results[algorithm] = results[results["algorithm"] == algorithm]

    return algorithm_results


if __name__ == "__main__":
    optimizer_data = read_optimizer_data_per_algorithm()
    algorithm = "ppo"
    overall_data = optimizer_data[algorithm]

    overall_data["optimizer_rank"] = overall_data.groupby(["algorithm", "environment"])["score"].rank(ascending=False)

    subset_data = overall_data[overall_data["environment"].isin(SUBSETS[algorithm])]

    subset_avg_rank = subset_data.groupby("optimizer")["optimizer_rank"].mean()
    overall_avg_rank = overall_data.groupby("optimizer")["optimizer_rank"].mean()

    fig, axes = plt.subplots(ncols=2, nrows=1, figsize=(10, 3))

    for ax, data, title in zip(axes, [subset_data, overall_data], ["Subset", "All Environments"]):
        sns.boxplot(data=subset_data, x="optimizer", y="optimizer_rank", ax=ax)
        ax.set_title(title)

    plt.tight_layout()
    plt.show()

    plt.savefig("subset_validation/plots/subset_validation.png", dpi=500)
