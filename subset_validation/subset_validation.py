import os
import pandas as pd
import ast
from collections import defaultdict
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict


EXPERIMENT_TO_ENV = {
    "brax_halfcheetah": "halfcheetah",
    "procgen_heist_easy": "HeistEasy-v0",
    "minigrid_unlock": "MiniGrid-Unlock",
    "procgen_plunder_easy": "PlunderEasy-v0",
    "procgen_jumper_easy": "JumperEasy-v0",
    "mujoco_inverted_double_pendulum": "InvertedDoublePendulum-v4",
    "cc_cartpole": "CartPole-v1",
    "procgen_bossfight_easy": "BossfightEasy-v0",
    "procgen_climber_easy": "ClimberEasy-v0",
    "atari_breakout": "Breakout-v5",
    "brax_swimmer": "swimmer",
    "brax_walker2d": "walker2d",
    "procgen_maze_easy": "MazeEasy-v0",
    "brax_fast": "fast",
    "procgen_dodgeball_easy": "DodgeballEasy-v0",
    "cc_pendulum": "Pendulum-v1",
    "brax_inverted_double_pendulum": "inverted_double_pendulum",
    "brax_humanoid_standup": "humanoidstandup",
    "atari_pong": "Pong-v5",
    "minigrid_empty_random": "MiniGrid-EmptyRandom-5x5",
    "procgen_ninja_easy": "NinjaEasy-v0",
    "box2d_lunar_lander": "LunarLander-v2",
    "atari_battle_zone": "BattleZone-v5",
    "brax_pusher": "pusher",
    "procgen_starpilot_easy": "StarpilotEasy-v0",
    "mujoco_reacher": "Reacher-v4",
    "atari_double_dunk": "DoubleDunk-v5",
    "mujoco_humanoid_standup": "HumanoidStandup-v4",
    "procgen_leaper_easy": "LeaperEasy-v0",
    "atari_phoenix": "Phoenix-v5",
    "mujoco_ant": "Ant-v4",
    "mujoco_pusher": "Pusher-v4",
    "mujoco_hopper": "Hopper-v4",
    "mujoco_inverted_pendulum": "InvertedPendulum-v4",
    "procgen_bigfish_easy": "BigfishEasy-v0",
    "mujoco_swimmer": "Swimmer-v4",
    "brax_hopper": "hopper",
    "brax_ant": "ant",
    "atari_this_game": "NameThisGame-v5",
    "mujoco_halfcheetah": "HalfCheetah-v4",
    "mujoco_humanoid": "Humanoid-v4",
    "brax_reacher": "reacher",
    "procgen_miner_easy": "MinerEasy-v0",
    "procgen_chaser_easy": "ChaserEasy-v0",
    "cc_continuous_mountain_car": "MountainCarContinuous-v0",
    "minigrid_four_rooms": "MiniGrid-FourRooms",
    "brax_inverted_pendulum": "inverted_pendulum",
    "atari_qbert": "Qbert-v5",
    "cc_mountain_car": "MountainCar-v0",
    "procgen_coinrun_easy": "CoinrunEasy-v0",
    "box2d_lunar_lander_continuous": "LunarLanderContinuous-v2",
    "minigrid_door_key": "MiniGrid-DoorKey-5x5",
    "box2d_bipedal_walker": "BipedalWalker-v3",
    "cc_acrobot": "Acrobot-v1",
    "procgen_fruitbot_easy": "FruitbotEasy-v0",
    "mujoco_walker2d": "Walker2d-v4",
    "procgen_caveflyer_easy": "CaveflyerEasy-v0",
    "brax_humanoid": "humanoid",
}


RAW_SOBOL_RESULTS = "results_combined/sobol"

OPTIMIZER_RESULTS = {
    "SMAC_BO": "results/smac",
    "SMAC_MultiFidelity": "results/smac_mf",
    "RandomSearch": "results/rs",
    "PBT": "results/pbt",
}

OPTIMIZER_NAMES = {
    "SMAC_BO": "SMAC BO",
    "SMAC_MultiFidelity": "SMAC with HB",
    "RandomSearch": "RS",
    "PBT": "PBT",
}

OPTIMIZER_SEEDS = {
    "SMAC_BO": [0, 1, 2],
    "SMAC_MultiFidelity": [0, 1, 2],
    "RandomSearch": [42, 43, 44],
    "PBT": [0, 1, 2],
}

SUBSETS = {
    "ppo": ["LunarLander-v2", "halfcheetah", "BattleZone-v5", "MiniGrid-EmptyRandom-5x5", "MiniGrid-FourRooms-5x5"],
    "dqn": ["Acrobot-v1", "MiniGrid-DoorKey-5x5", "BattleZone-v5", "MiniGrid-FourRooms-5x5"],
    "sac": ["BipedalWalker-v3", "halfcheetah", "MountainCarContinuous-v0", "Pendulum-v1"],
}

def read_min_max_scores():
    min_score = defaultdict(dict)
    max_score = defaultdict(dict)

    for result_file in os.listdir(RAW_SOBOL_RESULTS):
        result_path = os.path.join(RAW_SOBOL_RESULTS, result_file)
        if os.path.isdir(result_path):
            continue

        result_raw = pd.read_csv(result_path)

        # We don't need the particular hyperparameter configuration
        result_filtered = result_raw[["run_id", "performance", "seed"]]

        # Match desired format
        result_filtered = result_filtered.rename(
            columns={"run_id": "Configuration", "performance": "Score", "seed": "Seed"}
        )

        # Modify score s.t. higher is better
        # During the data collection we tried to minimize the cost, i.e. the core
        result_filtered["Score"] *= -1

        # Extract properties from the file name of the result
        splitted_filename = result_file.replace(".csv", "").split("_")
        algorithm = splitted_filename[-1]
        environment = "_".join(splitted_filename[:-1])

        environment = EXPERIMENT_TO_ENV[environment]

        min_score[algorithm][environment] = result_filtered["Score"].min()
        max_score[algorithm][environment] = result_filtered["Score"].max()

    return min_score, max_score


def read_optimizer_results(optimizer: str) -> pd.DataFrame:
    scores = []
    path = OPTIMIZER_RESULTS[optimizer]

    for result_dir in os.listdir(path):
        if os.path.isfile(os.path.join(path, result_dir)):
            continue
        algorithm = result_dir.split("_")[0]
        environment = result_dir.split("_")[1]

        for seed in OPTIMIZER_SEEDS[optimizer]:
            incumbent_path = os.path.join(
                path, result_dir, str(seed), "42", "incumbent.json"
            )
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

            scores += [
                {
                    "algorithm": algorithm,
                    "environment": environment,
                    "score": score,
                    "optimizer": optimizer,
                    "seed": seed,
                }
            ]
    scores = pd.DataFrame(scores)
    scores.to_csv(os.path.join(path, "combined.csv"), index=False)
    scores = scores.groupby(["algorithm", "environment"])["score"].mean().reset_index()
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


def validate(algorithm: str, method: str = "rank"):
    min_scores, max_scores = read_min_max_scores()
    optimizer_data = read_optimizer_data_per_algorithm()
    overall_data = optimizer_data[algorithm]
    subset = SUBSETS[algorithm]

    # Rename columns
    overall_data["optimizer"] = overall_data["optimizer"].replace(OPTIMIZER_NAMES)
    subset_data = overall_data[overall_data["environment"].isin(subset)]

    if method == "rank":
        overall_data.loc[:, "normalized_score"] = overall_data.groupby(["algorithm", "environment"])[
            "score"
        ].rank(ascending=False)
        subset_data.loc[:, "normalized_score"] = subset_data.groupby(["algorithm", "environment"])[
            "score"
        ].rank(ascending=False)
    else:
        def min_max_normalize(row):
            min_score = min_scores[row["algorithm"]][row["environment"]]
            max_score = max_scores[row["algorithm"]][row["environment"]]
            normalized_score = (row["score"] - min_score) / (max_score - min_score)
            return normalized_score

        # Apply min-max normalization
        overall_data.loc[:, "normalized_score"] = overall_data.apply(min_max_normalize, axis=1)
        subset_data.loc[:, "normalized_score"] = subset_data.apply(min_max_normalize, axis=1)
    

    return overall_data, subset_data


def plot_subset_vs_overall(overall_data, subset_data, method: str):
    fig, axes = plt.subplots(1, 2, figsize=(9, 2.5), sharey=True)

    sns.boxplot(x='optimizer', y='normalized_score', data=subset_data, ax=axes[0], showmeans=True, meanline=True)
    axes[0].set_title('Subset')

    sns.boxplot(x='optimizer', y='normalized_score', data=overall_data, ax=axes[1], showmeans=True, meanline=True)
    axes[1].set_title('All Environments')

    for ax in axes:
        ax.set_ylabel('Normalized Score')
        ax.set_xlabel('Optimizer')

    plt.tight_layout()
    plt.savefig(f"subset_validation/plots/{method}_{algorithm}_comparison.png", dpi=500)

if __name__ == "__main__":
    for method in ["rank", "min_max"]:
        for algorithm in ["ppo", "dqn", "sac"]:
            overall_data, subset_data = validate(algorithm, method) 
            plot_subset_vs_overall(overall_data, subset_data, method)


