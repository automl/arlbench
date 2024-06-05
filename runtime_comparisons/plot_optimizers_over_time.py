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
    "rs": "Random Search",
    "smac": "SMAC",
    "smac_mf": "SMAC + HB",
    "pbt": "PBT"
}
OPT_PLOTS_DIR = "plots/optimizer_runs"
OPT_RESULTS_DIR = "results_combined/optimizer_runs"
STEP_SIZE = 100000

HUE_ORDER = ["Random Search", "SMAC", "SMAC + HB", "PBT"]

ENV_CATEGORIES = {
    "ppo": {
        "Atari": ["BattleZone-v5", "DoubleDunk-v5", "NameThisGame-v5", "Phoenix-v5", "QBert-v5"],
        "Box2D": ["LunarLander-v2", "LunarLanderContinuous-v2"],
        "Classic Control": ["Acrobot-v1", "CartPole-v1", "MountainCar-v0", "MountainCarContinuous-v0", "Pendulum-v1"],
        "XLand": ["MiniGrid-DoorKey-5x5", "MiniGrid-EmptyRandom-5x5", "MiniGrid-FourRooms", "MiniGrid-Unlock"],
        "Brax": ["ant", "halfcheetah", "hopper", "humanoid"]
    },
    "dqn": {
        "Atari": ["BattleZone-v5", "DoubleDunk-v5", "NameThisGame-v5", "Phoenix-v5", "QBert-v5"],
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

SUBSETS = {
    "ppo": ["LunarLander-v2", "halfcheetah", "BattleZone-v5", "MiniGrid-EmptyRandom-5x5", "MiniGrid-FourRooms"],
    "dqn": ["Acrobot-v1", "MiniGrid-DoorKey-5x5", "BattleZone-v5", "MiniGrid-FourRooms"],
    "sac": ["BipedalWalker-v3", "halfcheetah", "MountainCarContinuous-v0", "Pendulum-v1"],
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

        environment = EXPERIMENT_TO_ENV[environment]

        min_score[algorithm][environment] = result_filtered["Score"].min()
        max_score[algorithm][environment] = result_filtered["Score"].max()

    return min_score, max_score

def read_opt_data(exp: str):
    logging.info(f"Reading {exp}")
    all_data = []

    for opt in OPTIMIZERS:
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

    return pd.concat(all_data) if len(all_data) > 0 else None


def interpolate(opt_data: pd.DataFrame) -> pd.DataFrame:
    interpolated_opt_datas = []
    
    min_budget = opt_data[opt_data["optimizer"] == "rs"]["cum_budget"].min()
    max_budget = opt_data[opt_data["optimizer"] == "rs"]["cum_budget"].max()

    for (optimizer, seed), group_opt_data in opt_data.groupby(["optimizer", "seed"]):
        max_opt_budget = group_opt_data["cum_budget"].max()

        new_cum_budget = pd.Series(np.arange(int(min_budget), int(max_opt_budget) + 1, 100), name="cum_budget")
        df_merged = pd.DataFrame(new_cum_budget)

        interpolated_opt_data = pd.merge(df_merged, group_opt_data, on="cum_budget", how="left")
        interpolated_opt_data["performance"] = interpolated_opt_data["performance"].ffill()
        interpolated_opt_data = interpolated_opt_data[interpolated_opt_data["cum_budget"] <= max_budget + 1]


        interpolated_opt_data["seed"] = seed - 42 if optimizer == "rs" else seed
        interpolated_opt_data["optimizer"] = OPTIMIZER_NAMES[optimizer]
    
        interpolated_opt_datas.append(interpolated_opt_data)

    df = pd.concat(interpolated_opt_datas, ignore_index=True)
    interpolated_data = df[df["cum_budget"] % STEP_SIZE == 0]
    interpolated_data[interpolated_data["cum_budget"] <= max_budget]

    return interpolated_data


def get_incumbent(opt_data: pd.DataFrame, exp: str, method: str, min_scores = None, max_scores = None) -> pd.DataFrame:
    best_config_opt_data = (
        opt_data.groupby(["optimizer", "cum_budget", "seed"], as_index=False)
        .apply(lambda group: group.loc[group["performance"].idxmin()])
        .reset_index(drop=True))
    best_config_opt_data = best_config_opt_data.drop(columns=["run_id"])

    result_path = os.path.join(OPT_RESULTS_DIR, f"{exp}.csv")
    if os.path.isfile(result_path) and False:   # TODO remove debug fix
        interpolated_opt_data = pd.read_csv(result_path)
    else:
        interpolated_opt_data = interpolate(best_config_opt_data)
        interpolated_opt_data.to_csv(result_path, index=False)
    incumbent_opt_data = interpolated_opt_data.sort_values(by=["optimizer", "seed", "cum_budget"])
    incumbent_opt_data["incumbent"] = incumbent_opt_data.groupby(["optimizer", "seed"])["performance"].cummin()
    incumbent_opt_data = incumbent_opt_data[["cum_budget", "optimizer", "seed", "incumbent"]]

    incumbent_opt_data = incumbent_opt_data.dropna()

    algorithm, environment = exp.split("_")

    if method == "rank":
         incumbent_opt_data["rank"] = incumbent_opt_data.groupby(["cum_budget", "seed"])["incumbent"].rank()
    elif method == "regret":
        def min_max_normalize(row):
            min_score = min_scores[algorithm][environment]
            max_score = max_scores[algorithm][environment]
            normalized_score = (row["incumbent"] - min_score) / (max_score - min_score)
            return normalized_score

        incumbent_opt_data.loc[:, "regret"] = incumbent_opt_data.apply(min_max_normalize, axis=1)

    incumbent_opt_data = incumbent_opt_data.sort_values("cum_budget")
    
    # Clip the optimizer by the minimum observed across all optimizers
    max_cum_budget_per_optimizer = incumbent_opt_data.groupby("optimizer")["cum_budget"].max()
    incumbent_opt_data = incumbent_opt_data[incumbent_opt_data["cum_budget"] <= max_cum_budget_per_optimizer.min()]

    incumbent_opt_data = incumbent_opt_data.sort_values("cum_budget")

    return incumbent_opt_data

def plot_opt_over_time(exp: str, method: str):
    min_scores, max_scores = None, None
    if method == "regret":
        min_scores, max_scores = read_min_max_scores()

    opt_data = read_opt_data(exp)
    if opt_data is None:
        return

    algorithm, environment = exp.split("_")
    
    incumbent_opt_data = get_incumbent(opt_data, exp, method, min_scores=min_scores, max_scores=max_scores)

    fig = plt.figure(figsize=(4, 3))
    g = sns.lineplot(data=incumbent_opt_data, x="cum_budget", y=method, hue="optimizer", errorbar=("ci", 95), hue_order=HUE_ORDER, drawstyle='steps')
    g.set_title(f"{algorithm.upper()} on {environment}")
    g.set_ylabel("Rank" if method == "rank" else "Regret")
    g.set_xlabel("Steps")

    g.legend(loc='upper center', bbox_to_anchor=(0.5, -0.25), ncol=2, fancybox=False, shadow=False, frameon=False)

    plt.tight_layout(pad=0.2, rect=(0, -0.05, 1, 1))

    path = os.path.join(OPT_PLOTS_DIR, method, f"{exp}.png")
    plt.savefig(path, dpi=500)
    logging.info(f"Saved {path}")

    plt.close()

def plot_envs_opt_over_time(algorithm: str, envs: list[str], category_name: str, method: str):
    min_scores, max_scores = None, None
    if method == "regret":
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

    min_cum_budget = concat_opt_data['cum_budget'].min()
    max_cum_budget = concat_opt_data['cum_budget'].max()

    for opt_data in all_opt_data:
        opt_data['normalized_cum_budget'] = (opt_data['cum_budget'] - min_cum_budget) / (max_cum_budget - min_cum_budget)

    all_opt_data = pd.concat(all_opt_data)

    # Clip the optimizer by the minimum observed across all optimizers
    # max_cum_budget_per_optimizer = all_opt_data.groupby("optimizer")["cum_budget"].max()
    # all_opt_data = all_opt_data[all_opt_data["cum_budget"] <= max_cum_budget_per_optimizer.min()]
    
    fig = plt.figure(figsize=(4, 3))
    g = sns.lineplot(data=all_opt_data, x="normalized_cum_budget", y=method, hue="optimizer", errorbar=("ci", 95), hue_order=HUE_ORDER)
    g.set_title(f"{algorithm.upper()} on {category_name}")
    g.set_ylabel("Rank" if method == "rank" else "Regret")
    g.set_xlabel("Steps")

    g.legend(loc='upper center', bbox_to_anchor=(0.5, -0.25), ncol=2, fancybox=False, shadow=False, frameon=False)

    plt.tight_layout(pad=0.2, rect=(0, -0.05, 1, 1))

    path = os.path.join(OPT_PLOTS_DIR, method, "categories", f"{algorithm}_{category_name}.png")
    plt.savefig(path, dpi=500)
    logging.info(f"Saved {path}")
    plt.close()


def plot_3_opt_over_time(experiments: list[str]):
    fig, axes = plt.subplots(ncols=len(experiments), nrows=1, figsize=(9, 2.5))

    hue_order = ["Random Search", "SMAC BO", "SMAC MF", "PBT"]

    for i, exp in enumerate(experiments):
        opt_data = read_opt_data(exp)

        algorithm, environment = exp.split("_")
            
        best_config_opt_data = opt_data.groupby(["optimizer", "cum_budget", "seed"]).apply(lambda x: x.loc[x["performance"].idxmin()]).reset_index(drop=True)

        interpolated_opt_data = interpolate(best_config_opt_data)

        incumbent_opt_data = interpolated_opt_data.sort_values(by=["optimizer", "seed", "cum_budget"])
        incumbent_opt_data["incumbent"] = incumbent_opt_data.groupby(["optimizer", "seed"])["performance"].cummin()

        sns.lineplot(data=incumbent_opt_data, x="cum_budget", y="incumbent", hue="optimizer", errorbar="sd", ax=axes[i], hue_order=hue_order)
        
        axes[i].set_title(f"{algorithm.upper()} on {environment}")
        axes[i].set_ylabel("")
        axes[i].set_xlabel("budget")
        axes[i].legend().set_visible(False)

    axes[0].set_ylabel("Cost")

    plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.1), ncol=4, fancybox=False, shadow=False, frameon=False)

    plt.tight_layout(pad=1.3)
    path = os.path.join(OPT_PLOTS_DIR, f"three_opt.png")
    plt.savefig(path, dpi=500)
    plt.close()


if __name__ == "__main__":
    # plot_envs_opt_over_time("dqn", ["CartPole-v1", "Acrobot-v1"], "test", "regret")

    for algorithm, subset_envs in SUBSETS.items():
        plot_envs_opt_over_time(algorithm, subset_envs, "Subset", "regret")
        break

    for algorithm, category in ENV_CATEGORIES.items():
        envs = [env for cat_envs in category.values() for env in cat_envs]
        plot_envs_opt_over_time(algorithm, envs, "Full Set", "regret")
        break

    for algorithm, category in ENV_CATEGORIES.items():
        for category_name, envs in category.items():
            plot_envs_opt_over_time(algorithm, envs, category_name, "regret")
            break

    for exp in os.listdir("results/smac_mf"):
       plot_opt_over_time(exp, "rank")
       plot_opt_over_time(exp, "regret")
       gc.collect()
       break