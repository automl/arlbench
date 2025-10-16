import pandas as pd
from pathlib import Path
from collections import defaultdict
from scipy.stats import spearmanr
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style("whitegrid")
sns.set_palette("colorblind")

DATA_DIR = Path("results_combined/sobol")
PLOTS_DIR = Path("plots/subset_validation")

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
    "box2d_continuous_lunar_lander": "LunarLanderContinuous-v2",
    "box2d_lunar_lander_continuous": "LunarLanderContinuous-v2",
    "minigrid_door_key": "MiniGrid-DoorKey-5x5",
    "box2d_bipedal_walker": "BipedalWalker-v3",
    "cc_acrobot": "Acrobot-v1",
    "procgen_fruitbot_easy": "FruitbotEasy-v0",
    "mujoco_walker2d": "Walker2d-v4",
    "procgen_caveflyer_easy": "CaveflyerEasy-v0",
    "brax_humanoid": "humanoid",
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
        "BipedalWalker-v3": 0.3208448,
        "halfcheetah": 0.317615,
        "hopper": 0.15381655,
        "MountainCarContinuous-v0": 0.19360028,
    },
}


def read_all_data() -> dict[str, pd.DataFrame]:
    all_data = defaultdict(list)
    
    for file in DATA_DIR.glob("*.csv"):
        filename = file.stem

        algorithm = filename.split("_")[-1]
        env_name = "_".join(filename.split("_")[:-1])
        try:
            env_name = EXPERIMENT_TO_ENV[env_name]
        except KeyError:
            print(f"Unknown env name: {env_name}")
            continue

        df = pd.read_csv(file)
        df = df[["run_id", "budget", "performance", "seed"]]
        df = df.rename(columns={"run_id": "config_id"})

        # Normalize budget 
        df.budget = df.budget / df.budget.max()

        # Min-max normalize performance
        df["performance"] = df["performance"].fillna(df["performance"].min())
        df["performance"] = (df["performance"] - df["performance"].min()) / (df["performance"].max() - df["performance"].min())

        # Mean over seeds
        df = df.groupby(["config_id", "budget"]).performance.mean().reset_index()
        df["env"] = env_name
        df["algorithm"] = algorithm

        all_data[algorithm] += [df]

    return {alg: pd.concat(dfs, ignore_index=True) for alg, dfs in all_data.items()}

def get_subset(df: pd.DataFrame, algorithm: str) -> pd.DataFrame:
    envs = SUBSET_WEIGHTS[algorithm].keys()
    subset_df = df[df["env"].isin(envs)].copy()
    return subset_df

def get_temporal_correlation(full_set: pd.DataFrame, algorithm: str, n_bootstraps: int = 100):
    subset = get_subset(full_set, algorithm)

    # Compute mean per config_id and budget for both sets
    full_set_mean = full_set.groupby(["config_id", "budget"]).performance.mean().reset_index()
    subset_mean = subset.groupby(["config_id", "budget"]).performance.mean().reset_index()

    budgets = subset_mean["budget"].unique()

    # Store results
    results = []

    for budget in budgets:
        subset_perf = subset_mean[subset_mean["budget"] == budget]['performance'].values
        fullset_perf = full_set_mean[full_set_mean['budget'] == budget]['performance'].values

        # Original correlation
        corr, p_value = spearmanr(subset_perf, fullset_perf)

        # Bootstrapping
        boot_corrs = []
        n = len(subset_perf)
        for _ in range(n_bootstraps):
            indices = np.random.choice(range(n), size=n, replace=True)
            boot_subset = subset_perf[indices]
            boot_fullset = fullset_perf[indices]
            boot_corr, _ = spearmanr(boot_subset, boot_fullset)
            boot_corrs.append(boot_corr)

        # Confidence interval (e.g., 95%)
        ci_lower = np.percentile(boot_corrs, 5)
        ci_upper = np.percentile(boot_corrs, 95)

        results.append({
            'budget': budget,
            'spearman_corr': corr,
            'p_value': p_value,
            'ci_lower': ci_lower,
            'ci_upper': ci_upper,
            'std_dev': np.std(boot_corrs)
        })

    results_df = pd.DataFrame(results)

    return results_df

def plot_temporal_correlation(all_data: dict[str, pd.DataFrame]):
    fig, axs = plt.subplots(1, 3, figsize=(9.5, 2.5), sharey=True)
    for algorithm, ax in zip(all_data.keys(), axs):
        results = get_temporal_correlation(all_data[algorithm], algorithm)

        sns.lineplot(data=results, x='budget', y='spearman_corr', ax=ax, marker='o')
        ax.fill_between(results['budget'], results['ci_lower'], results['ci_upper'], alpha=0.3)
        ax.set_title(f'{algorithm.upper()}')
        ax.set_xlabel('Normalized Training Steps')
        ax.set_ylabel('Spearman Correlation')
        ax.set_ylim(0, 1)
        ax.grid(True)
    
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "temporal_correlation.png", dpi=500)

if __name__ == "__main__":
    np.random.seed(42)

    all_data = read_all_data()
    plot_temporal_correlation(all_data)




        