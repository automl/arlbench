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

# pd.set_option('display.max_rows', None)


logging.basicConfig(stream=sys.stdout, level=logging.INFO, format="%(asctime)s %(message)s")

warnings.filterwarnings("ignore")

N_BUCKETS = 1000

RAW_SOBOL_RESULTS = "results_combined/sobol"

N_CONFIGS = {
    "rs": 32,
    "smac": 32,
    "smac_mf": 70,
    "pbt": 320
}

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
        "ALE": ["BattleZone-v5", "DoubleDunk-v5", "NameThisGame-v5", "Phoenix-v5", "Qbert-v5"],
        "Box2D": ["LunarLander-v2", "LunarLanderContinuous-v2"],
        "Classic Control": ["Acrobot-v1", "CartPole-v1", "MountainCar-v0", "MountainCarContinuous-v0", "Pendulum-v1"],
        "XLand": ["MiniGrid-DoorKey-5x5", "MiniGrid-EmptyRandom-5x5", "MiniGrid-FourRooms", "MiniGrid-Unlock"],
        "Brax": ["ant", "halfcheetah", "hopper", "humanoid"]
    },
    "dqn": {
        "ALE": ["BattleZone-v5", "DoubleDunk-v5", "NameThisGame-v5", "Phoenix-v5", "Qbert-v5"],
        "Box2D": ["LunarLander-v2"],
        "Classic Control": ["Acrobot-v1", "CartPole-v1", "MountainCar-v0"],
        "XLand": ["MiniGrid-DoorKey-5x5", "MiniGrid-EmptyRandom-5x5", "MiniGrid-FourRooms", "MiniGrid-Unlock"],
    },
    "sac": {
        "Box2D": ["BipedalWalker-v3", "LunarLanderContinuous-v2"],
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

        # In the landscaping we minimized the score, i.e. negative rewards
        result_filtered["Score"] *= -1

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

def read_incubment_data(exp: str):
    environment = exp.split("_")[1]

    all_data = []

    for opt in OPTIMIZERS:
        logging.info(f"Reading {exp}: {opt}")
        exp_path = os.path.join("results", opt, exp)
        if not os.path.isdir(exp_path):
            print(f"Unable to load {exp_path}")
            continue 

        for seed_dir in os.listdir(exp_path):
            seed_path = os.path.join(exp_path, seed_dir)

            if not os.path.isdir(seed_path):
                continue

            incumbent_path = os.path.join(seed_path, "42", "incumbent.csv")
            runhistory_path = os.path.join(seed_path, "42", "runhistory.csv")

            if not os.path.isfile(incumbent_path):
                print(f"Unable to read directory {seed_path}.")
                continue 

            incumbent_data = pd.read_csv(incumbent_path)
            runhistory_data = pd.read_csv(runhistory_path)

            if opt == "pbt":
                # Sometimes the hypersweeper excutes one additional run at the 
                # end so we have to remove it 
                runhistory_data = runhistory_data.iloc[:32 * 10, :]
                incumbent_data = incumbent_data.iloc[:10, :]

                # To fix the used budget, we determine the budget of one iteration
                # and compute the cumulative sum
                run_budget = float(runhistory_data["budget"].values[0])
                try:
                    incumbent_data["budget_used"] = np.array([run_budget * 32] * 10).cumsum()
                except:
                    continue

            if not len(runhistory_data) == N_CONFIGS[opt]:
                print(f"{runhistory_path}: Not finished.")
                continue

            incumbent_data["norm_budget"] = incumbent_data["budget_used"] / (incumbent_data["budget_used"].max() / 32)

            all_norm_budget = pd.DataFrame({"norm_budget": np.arange(0, 32.025, 0.025)})
            incumbent_data = pd.merge(all_norm_budget, incumbent_data, on="norm_budget", how="left")

            incumbent_data["performance"] = incumbent_data["performance"].ffill()

            incumbent_data = incumbent_data[["norm_budget", "performance"]]

            incumbent_data["optimizer"] = OPTIMIZER_NAMES[opt]
            incumbent_data["environment"] = environment
            incumbent_data["seed"] = int(seed_dir) - 42 if opt == "rs" else int(seed_dir)

            all_data.append(incumbent_data)

    all_df =  pd.concat(all_data) if len(all_data) > 0 else None

    return all_df


def normalize(
        incumbent_data: pd.DataFrame, 
        exp: str, 
        method: str, 
        min_scores: dict | None = None, 
        max_scores: dict | None = None
    ) -> pd.DataFrame:    
    algorithm, environment = exp.split("_")

    if method == "rank":
        incumbent_data["rank"] = incumbent_data.groupby(["norm_budget", "seed", "environment"])["performance"].rank(ascending=True)
    elif method == "score":
        assert min_scores is not None
        assert max_scores is not None

        def min_max_normalize(row):
            overall_min = incumbent_data["performance"].min()
            overall_max = incumbent_data["performance"].max()

            min_score = min(min_scores[algorithm][environment], overall_min)
            max_score = max(max_scores[algorithm][environment], overall_max)
            normalized_score = (row["performance"] - min_score) / (max_score - min_score)
            return normalized_score

        incumbent_data.loc[:, "score"] = incumbent_data.apply(min_max_normalize, axis=1)
        # incumbent_data.loc[:, "score"] *= -1
        # incumbent_data.loc[:, "score"] -= 1
        # incumbent_data.loc[:, "score"] = incumbent_data.loc[:, "score"].fillna(0)

    return incumbent_data


def plot_opt_over_time(exp: str, method: str):
    min_scores, max_scores = None, None
    if method == "score":
        min_scores, max_scores = read_min_max_scores()

    incumbent_data = read_incubment_data(exp)
    if incumbent_data is None:
        return
    
    incumbent_data = normalize(
        incumbent_data=incumbent_data,
        exp=exp,
        method=method,
        min_scores=min_scores,
        max_scores=max_scores
    )

    incumbent_data = incumbent_data.reset_index()

    algorithm, environment = exp.split("_")

    for scale in ["", "log"]:
        fig = plt.figure(figsize=(4, 3))
        g = sns.lineplot(
            data=incumbent_data,
            x="norm_budget", 
            y=method, 
            hue="optimizer", 
            errorbar=("ci", 95), 
            hue_order=HUE_ORDER, 
            drawstyle="steps")
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
        incumbent_data = read_incubment_data(exp)
        if incumbent_data is None:
            continue

        incumbent_data = normalize(
            incumbent_data=incumbent_data,
            exp=exp,
            method=method,
            min_scores=min_scores,
            max_scores=max_scores
        )

        all_opt_data += [incumbent_data]

    if len (all_opt_data) == 0:
        return

    concat_opt_data = pd.concat(all_opt_data)
    concat_opt_data = concat_opt_data.reset_index()

    for scale in ["", "log"]:
        fig = plt.figure(figsize=(4, 3))
        g = sns.lineplot(data=concat_opt_data, x="norm_budget", y=method, hue="optimizer", errorbar=("ci", 95), hue_order=HUE_ORDER, drawstyle='steps')
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


def plot_subset_vs_overall(algorithm: str, method: str):
    min_scores, max_scores = read_min_max_scores()

    subset_envs = list(SUBSET_WEIGHTS[algorithm].keys())
    overall_envs = [env for cat_envs in ENV_CATEGORIES[algorithm].values() for env in cat_envs]

    subset_data, overall_data = [], []
    for envs, data in zip([subset_envs, overall_envs], [subset_data, overall_data]):
        for env in envs:
            exp = f"{algorithm}_{env}"
            incumbent_data = read_incubment_data(exp)
            if incumbent_data is None:
                continue

            # incumbent_opt_data = get_incumbent(opt_data, exp, "score", min_scores=min_scores, max_scores=max_scores)
            incumbent_data = normalize(
                incumbent_data=incumbent_data,
                exp=exp,
                method=method,
                min_scores=min_scores,
                max_scores=max_scores
            )

            data.append(incumbent_data)

        if len (data) == 0:
            return

    subset_data = pd.concat(subset_data)
    overall_data = pd.concat(overall_data)
    # subset_data = subset_data.reset_index()
    # overall_data = overall_data.reset_index()

    # Compute mean over all envs but keep seeds
    subset_data = subset_data.groupby(["norm_budget", "optimizer", "seed"])["score"].mean().reset_index()
    overall_data = overall_data.groupby(["norm_budget", "optimizer", "seed"])["score"].mean().reset_index()

    for scale in ["", "log"]:
        fig, axes = plt.subplots(ncols=2, nrows=1, figsize=(9, 2.5), sharey=True)

        for data, ax, set_name in zip([subset_data, overall_data], axes, ["Subset", "All"]):
            g = sns.lineplot(
                data=data,
                x="norm_budget",
                y=method,
                hue="optimizer",
                errorbar=("ci", 95),
                hue_order=HUE_ORDER,
                drawstyle='steps',
                ax=ax
            )
            g.set_title(f"{algorithm.upper()}: {set_name}")
            g.set_ylabel("Normalized Score")
            g.set_xlabel("Full RL Trainings")

            g.set_ylim(None, 1)

            if scale == "log":
                g.set_xlim(0.125, 32)
                g.set_xscale("log")
            else:
                g.set_xlim(0, 32)
                g.set_xticks([0, 4, 8, 12, 16, 20, 24, 28, 32])

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
        plot_subset_vs_overall(algorithm, "score")


    # for algorithm, subset_weights in SUBSET_WEIGHTS.items():
    #     subset_envs = list(subset_weights.keys())
    #     plot_envs_opt_over_time(algorithm, subset_envs, "Subset", "rank")

    # for algorithm, category in ENV_CATEGORIES.items():
    #     envs = [env for cat_envs in category.values() for env in cat_envs]
    #     plot_envs_opt_over_time(algorithm, envs, "Full Set", "score")

    # for algorithm, category in ENV_CATEGORIES.items():
    #     for category_name, envs in category.items():
    #         plot_envs_opt_over_time(algorithm, envs, category_name, "rank")

    #plot_envs_opt_over_time("ppo", ENV_CATEGORIES["ppo"]["Classic Control"], "Classic Control", "rank")

    # plot_opt_over_time("ppo_LunarLander-v2", "score")

    # for exp in os.listdir("results/rs"):
    #    plot_opt_over_time(exp, "score")
    #    gc.collect()
