import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import os
from scipy.interpolate import interp1d
import numpy as np
import gc
from collections import defaultdict
import warnings
import logging
import sys


logging.basicConfig(stream=sys.stdout, level=logging.INFO, format="%(asctime)s %(message)s")

warnings.filterwarnings("ignore")

N_BUCKETS = 1000

RAW_SOBOL_RESULTS = "results_combined/sobol"

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

OPTIMIZERS = ["rs", "smac", "smac_mf", "pbt"]
OPTIMIZER_NAMES = {
    "rs": "RS",
    "smac": "SMAC",
    "smac_mf": "SMAC + HB",
    "pbt": "PBT"
}
OPT_PLOTS_DIR = "plots/subset_validation/optimizer_runs"
OPT_RESULTS_DIR = "results_combined/optimizer_runs"

HUE_ORDER = ["RS", "SMAC", "SMAC + HB", "PBT"]

ENV_CATEGORIES = {
    "ppo": {
        "ALE": ["BattleZone-v5", "DoubleDunk-v5", "NameThisGame-v5", "Phoenix-v5", "QBert-v5"],
        "Box2D": ["LunarLander-v2", "LunarLanderContinuous-v2"],
        "Classic Control": ["Acrobot-v1", "CartPole-v1", "MountainCar-v0", "MountainCarContinuous-v0", "Pendulum-v1"],
        "XLand": ["MiniGrid-DoorKey-5x5", "MiniGrid-EmptyRandom-5x5", "MiniGrid-FourRooms", "MiniGrid-Unlock"],
        "Brax": ["ant", "halfcheetah", "hopper", "humanoid"]
    },
    "dqn": {
        "ALE": ["BattleZone-v5", "DoubleDunk-v5", "NameThisGame-v5", "Phoenix-v5", "QBert-v5"],
        "Box2D": ["LunarLander-v2"],
        "Classic Control": ["Acrobot-v1", "CartPole-v1", "MountainCar-v0"],
        "XLand": ["MiniGrid-DoorKey-5x5", "MiniGrid-EmptyRandom-5x5", "MiniGrid-FourRooms", "MiniGrid-Unlock"],
    },
    "sac": {
        "Box2D": ["BipedalWalker-v2", "LunarLanderContinuous-v2"],
        "Classic Control": ["MountainCarContinuous-v0", "Pendulum-v1"],
        "Brax": ["ant", "halfcheetah", "hopper", "humanoid"]
    },
}

SUBSET_WEIGHTS = {
    "ppo": {
        "BattleZone-v5": 0.18960638,
        "Phoenix-v5": 0.12810087,
        "LunarLander-v2": 0.21154265,
        "humanoid": 0.21554603,
        "MiniGrid-EmptyRandom-5x5": 0.23619909
    },
    "dqn": {
        "DoubleDunk-v5": 0.22108139,
        "NameThisGame-v5": 0.10913745,
        "Acrobot-v1": 0.3300676,
        "MiniGrid-EmptyRandom-5x5": 0.18383447,
        "MiniGrid-FourRooms": 0.11920235,
    },
    "sac": {
        "ant": 0.35797678,
        "halfcheetah": 0.3176157,
        "hopper": 0.15381655,
        "MountainCarContinuous-v0": 0.19360028,
    },
}


sns.set_style("whitegrid")
sns.set_palette("colorblind")


def read_min_max_scores():
    min_score = defaultdict(dict)
    max_score = defaultdict(dict)

    for result_file in os.listdir(RAW_SOBOL_RESULTS):
        result_path = os.path.join(RAW_SOBOL_RESULTS, result_file)
        if os.path.isdir(result_path):
            continue

        result_raw = pd.read_csv(result_path)

        # We don"t need the particular hyperparameter configuration
        result_filtered = result_raw[["run_id", "performance", "seed"]]

        # Match desired format
        result_filtered = result_filtered.rename(
            columns={"run_id": "Configuration", "performance": "Score", "seed": "Seed"}
        )

        # Extract properties from the file name of the result
        splitted_filename = result_file.replace(".csv", "").split("_")
        algorithm = splitted_filename[-1]
        environment = "_".join(splitted_filename[:-1])

        min_p = result_filtered.dropna()["Score"].min()
        max_p = result_filtered.dropna()["Score"].max()

        if "brax" in environment:
            min_p = -2000
        elif "box2d" in environment:
            min_p = -200

        environment = EXPERIMENT_TO_ENV[environment]

        min_score[algorithm][environment] = min_p
        max_score[algorithm][environment] = max_p

    return min_score, max_score

def read_opt_data(exp: str):
    all_data = []

    for opt in OPTIMIZERS:
        logging.info(f"Reading {exp}: {opt}")
        runhistory_path = os.path.join("results", opt, exp, "runhistory_combined.csv")

        if not os.path.isfile(runhistory_path):
            continue

        opt_data = pd.read_csv(runhistory_path)
        opt_data["optimizer"] = opt
        opt_data = opt_data[["optimizer", "run_id", "budget", "performance", "seed"]]

        if opt == "smac_mf" or opt == "pbt":
            agg_opt_data = []

            for (budget, seed), budget_seed_group in opt_data.groupby(["budget", "seed"]):
                if opt == "pbt":
                    budget = opt_data["budget"].min()
                agg_opt_data += [{
                    "budget": budget,
                    "cum_budget": budget * len(budget_seed_group),
                    "seed": seed,
                    "performance": budget_seed_group["performance"].min()
                }]


            agg_opt_data = pd.DataFrame(agg_opt_data)

            agg_opt_data["optimizer"] = opt
            agg_opt_data = agg_opt_data.sort_values("budget")
            agg_opt_data["cum_budget"] = agg_opt_data.groupby("seed")["cum_budget"].cumsum()
            
            all_data += [agg_opt_data]
        else:
            opt_data["cum_budget"] = (opt_data["run_id"] + 1) * opt_data["budget"].min()
            all_data += [opt_data]

    all_df =  pd.concat(all_data) if len(all_data) > 0 else None

    return all_df

def interpolate(opt_data: pd.DataFrame) -> pd.DataFrame:
    interpolated_opt_datas = []
    
    min_budget = opt_data[opt_data["optimizer"] == "rs"]["cum_budget"].min()
    max_budget = opt_data[opt_data["optimizer"] == "rs"]["cum_budget"].max()

    for (optimizer, seed), group_opt_data in opt_data.groupby(["optimizer", "seed"]):
        group_opt_data = group_opt_data.drop(columns=["budget"])
        group_opt_data["performance"] = group_opt_data["performance"].ffill()

        group_opt_data = group_opt_data[group_opt_data["cum_budget"] <= max_budget]

        group_opt_data = group_opt_data.dropna(subset=["performance"])

        group_opt_data["norm_budget"] = (N_BUCKETS * group_opt_data["cum_budget"] / max_budget).astype(int)
        
        group_opt_data = group_opt_data.loc[group_opt_data.groupby('norm_budget')['performance'].idxmin()]

        all_norm_budget = pd.DataFrame({'norm_budget': range(0, N_BUCKETS + 1)})

        all_norm_budget = pd.merge(all_norm_budget, group_opt_data, on='norm_budget', how='left')

        all_norm_budget["performance"] = all_norm_budget["performance"].ffill()
        all_norm_budget["seed"] = seed - 42 if optimizer == "rs" else seed
        all_norm_budget["optimizer"] = OPTIMIZER_NAMES[optimizer]
        all_norm_budget["norm_budget"] /= N_BUCKETS

        interpolated_opt_datas.append(all_norm_budget)

    interpolated_opt_datas = pd.concat(interpolated_opt_datas, ignore_index=True)
    interpolated_opt_datas[interpolated_opt_datas["cum_budget"] <= max_budget]

    return interpolated_opt_datas


def get_incumbent(opt_data: pd.DataFrame, exp: str, method: str, min_scores = None, max_scores = None) -> pd.DataFrame:
    best_config_opt_data = (
        opt_data.groupby(["optimizer", "cum_budget", "seed"], as_index=False)
        .apply(lambda group: group.loc[group["performance"].idxmin()])
        .reset_index(drop=True))
    best_config_opt_data = best_config_opt_data.drop(columns=["run_id"])

    interpolated_opt_data = interpolate(best_config_opt_data)
    
    incumbent_opt_data = interpolated_opt_data.sort_values(by=["optimizer", "seed", "norm_budget"])
    incumbent_opt_data["incumbent"] = incumbent_opt_data.groupby(["optimizer", "seed"])["performance"].cummin()
    incumbent_opt_data = incumbent_opt_data[["cum_budget", "norm_budget", "optimizer", "seed", "incumbent"]]

    algorithm, environment = exp.split("_")

    if method == "rank":
         incumbent_opt_data["rank"] = incumbent_opt_data.groupby(["norm_budget", "seed"])["incumbent"].rank()
    elif method == "score":
        def min_max_normalize(row):
            min_score = min_scores[algorithm][environment]
            max_score = max_scores[algorithm][environment]
            normalized_score = (row["incumbent"] - min_score) / (max_score - min_score)
            return max(min(normalized_score, 1), 0)

        incumbent_opt_data.loc[:, "score"] = incumbent_opt_data.apply(min_max_normalize, axis=1)
        incumbent_opt_data.loc[:, "score"] *= -1
        incumbent_opt_data.loc[:, "score"] += 1
        incumbent_opt_data.loc[:, "score"] = incumbent_opt_data.loc[:, "score"].fillna(0)


    return incumbent_opt_data

def plot_opt_over_time(exp: str, method: str):
    min_scores, max_scores = None, None
    if method == "score":
        min_scores, max_scores = read_min_max_scores()

    opt_data = read_opt_data(exp)
    if opt_data is None:
        return

    algorithm, environment = exp.split("_")
    
    incumbent_opt_data = get_incumbent(opt_data, exp, method, min_scores=min_scores, max_scores=max_scores)

    for scale in ["", "log"]:
        fig = plt.figure(figsize=(4, 3))
        g = sns.lineplot(data=incumbent_opt_data, x="norm_budget", y=method, hue="optimizer", errorbar=("ci", 95), hue_order=HUE_ORDER, drawstyle='steps')
        g.set_title(f"{algorithm.upper()} on {environment}")
        g.set_ylabel("Rank" if method == "rank" else "Normalized Score")
        g.set_xlabel("Normalized Steps")

        if scale == "log":
            g.set_xscale("log")

        g.legend(loc='upper center', bbox_to_anchor=(0.5, -0.25), ncol=2, fancybox=False, shadow=False, frameon=False)

        plt.tight_layout(rect=(0, -0.05, 1, 1))

        file_name = f"{algorithm}_{environment}_{method}"
        if scale == "log":
            file_name += "_log"

        path = os.path.join(OPT_PLOTS_DIR, "single_runs", f"{file_name}.png")
        plt.savefig(path, dpi=500)
        logging.info(f"Saved {path}")
        plt.close()


def plot_envs_opt_over_time(algorithm: str, envs: list[str], category_name: str, method: str):
    min_scores, max_scores = None, None
    if method == "score":
        min_scores, max_scores = read_min_max_scores()

    all_opt_data = []
    for env in envs:
        exp = f"{algorithm}_{env}"
        opt_data = read_opt_data(exp)
        if opt_data is None:
            continue

        incumbent_opt_data = get_incumbent(opt_data, exp, method, min_scores=min_scores, max_scores=max_scores)

        incumbent_opt_data["env"] = env
        all_opt_data += [incumbent_opt_data]

    if len (all_opt_data) == 0:
        return

    concat_opt_data = pd.concat(all_opt_data)
    concat_opt_data = concat_opt_data.reset_index()
    
    for scale in ["", "log"]:
        fig = plt.figure(figsize=(4, 3))
        g = sns.lineplot(data=concat_opt_data, x="norm_budget", y="score", hue="optimizer", errorbar=("ci", 95), hue_order=HUE_ORDER, drawstyle='steps')
        g.set_title(f"{algorithm.upper()} on {category_name}")
        g.set_ylabel("Rank" if method == "rank" else "Normalized Score")
        g.set_xlabel("Normalized Steps")

        if scale == "log":
            g.set_xscale("log")

        g.legend(loc='upper center', bbox_to_anchor=(0.5, -0.25), ncol=2, fancybox=False, shadow=False, frameon=False)

        plt.tight_layout(rect=(0, -0.05, 1, 1))

        file_name = f"{algorithm}_{category_name}_{method}"
        if scale == "log":
            file_name += "_log"

        path = os.path.join(OPT_PLOTS_DIR, "subsets", f"{file_name}.png")
        plt.savefig(path, dpi=500)
        logging.info(f"Saved {path}")
        plt.close()


def plot_subset_vs_overall(algorithm: str):
    min_scores, max_scores = read_min_max_scores()

    subset_envs = list(SUBSET_WEIGHTS[algorithm].keys())
    overall_envs = [env for cat_envs in ENV_CATEGORIES[algorithm].values() for env in cat_envs]

    subset_data, overall_data = [], []
    for envs, data in zip([subset_envs, overall_envs], [subset_data, overall_data]):
        for env in envs:
            exp = f"{algorithm}_{env}"
            opt_data = read_opt_data(exp)
            if opt_data is None:
                continue

            incumbent_opt_data = get_incumbent(opt_data, exp, "score", min_scores=min_scores, max_scores=max_scores)

            incumbent_opt_data["env"] = env
            data.append(incumbent_opt_data)

        if len (data) == 0:
            return

    subset_data = pd.concat(subset_data)
    overall_data = pd.concat(overall_data)
    subset_data = subset_data.reset_index()
    overall_data = overall_data.reset_index()

    print(subset_data)
    
    for scale in ["", "log"]:
        fig, axes = plt.subplots(ncols=2, nrows=1, figsize=(9, 2.5), sharey=True)

        for data, ax, set_name in zip([subset_data, overall_data], axes, ["Subset", "All"]):
            g = sns.lineplot(data=data, x="norm_budget", y="score", hue="optimizer", errorbar=("ci", 95), hue_order=HUE_ORDER, drawstyle='steps', ax=ax)
            g.set_title(f"{algorithm.upper()}: {set_name}")
            g.set_ylabel("Normalized Score")
            g.set_xlabel("Normalized Steps")
            g.set_ylim(0, 1)

            if scale == "log":
                g.set_xscale("log")

        legend = axes[0].legend()
        for ax in axes:
            ax.legend().set_visible(False)
        fig.legend(legend.legend_handles, [t.get_text() for t in legend.texts], loc='upper center', bbox_to_anchor=(0.5, 0.1), ncol=4, fancybox=False, shadow=False, frameon=False)

        plt.tight_layout(rect=(0, 0.07, 1, 1))

        file_name = f"{algorithm}_subset_vs_overall"
        if scale == "log":
            file_name += "_log"

        path = os.path.join(OPT_PLOTS_DIR, f"{file_name}.png")
        plt.savefig(path, dpi=500)
        logging.info(f"Saved {path}")
        plt.close()


if __name__ == "__main__":
    for algorithm in SUBSET_WEIGHTS:
        plot_subset_vs_overall(algorithm)

    # for algorithm, subset_weights in SUBSET_WEIGHTS.items():
    #     subset_envs = list(subset_weights.keys())
    #     plot_envs_opt_over_time(algorithm, subset_envs, "Subset", "score")

    # for algorithm, category in ENV_CATEGORIES.items():
    #     envs = [env for cat_envs in category.values() for env in cat_envs]
    #     plot_envs_opt_over_time(algorithm, envs, "Full Set", "score")

    for algorithm, category in ENV_CATEGORIES.items():
        for category_name, envs in category.items():
            plot_envs_opt_over_time(algorithm, envs, category_name, "score")

    for exp in os.listdir("results/rs"):
       plot_opt_over_time(exp, "score")
       gc.collect()
