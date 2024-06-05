import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import os
from scipy.interpolate import interp1d
import numpy as np
import gc


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


def read_opt_data(exp: str):
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

        df_merged = pd.merge(df_merged, group_opt_data, on="cum_budget", how="left")
        interpolated_opt_data = df_merged.interpolate(method="linear")
        interpolated_opt_data = interpolated_opt_data[interpolated_opt_data["cum_budget"] <= max_budget + 1]


        interpolated_opt_data["seed"] = seed - 42 if optimizer == "rs" else seed
        interpolated_opt_data["optimizer"] = OPTIMIZER_NAMES[optimizer]
    
        interpolated_opt_datas.append(interpolated_opt_data)

    df = pd.concat(interpolated_opt_datas, ignore_index=True)
    interpolated_data = df[df["cum_budget"] % STEP_SIZE == 0]
    interpolated_data[interpolated_data["cum_budget"] <= max_budget]

    del df

    return interpolated_data


def get_incumbent(opt_data: pd.DataFrame, exp: str, method: str) -> pd.DataFrame:
    best_config_opt_data = opt_data.groupby(["optimizer", "cum_budget", "seed"]).apply(lambda x: x.loc[x["performance"].idxmin()]).reset_index(drop=True)
    
    result_path = os.path.join(OPT_RESULTS_DIR, f"{exp}.csv")
    if os.path.isfile(result_path):
        interpolated_opt_data = pd.read_csv(result_path)
    else:
        interpolated_opt_data = interpolate(best_config_opt_data)
        interpolated_opt_data.to_csv(result_path, index=False)

    incumbent_opt_data = interpolated_opt_data.sort_values(by=["optimizer", "seed", "cum_budget"])
    incumbent_opt_data["incumbent"] = incumbent_opt_data.groupby(["optimizer", "seed"])["performance"].cummin()
    incumbent_opt_data = incumbent_opt_data[["cum_budget", "optimizer", "seed", "incumbent"]]

    incumbent_opt_data = incumbent_opt_data.dropna()

    if method == "rank":
        incumbent_opt_data["rank"] = incumbent_opt_data.groupby(["cum_budget", "seed"])["incumbent"].rank()

    incumbent_opt_data = incumbent_opt_data.sort_values("cum_budget")
    return incumbent_opt_data

def plot_opt_over_time(exp: str, rank: bool = False):
    opt_data = read_opt_data(exp)
    if opt_data is None:
        return

    algorithm, environment = exp.split("_")
    
    method = "rank" if rank else ""
    incumbent_opt_data = get_incumbent(opt_data, exp, method)

    fig = plt.figure(figsize=(7, 5))
    g = sns.lineplot(data=incumbent_opt_data, x="cum_budget", y="rank" if rank else "incumbent", hue="optimizer", errorbar=("ci", 95), hue_order=HUE_ORDER)
    g.set_title(f"{algorithm.upper()} on {environment}")
    g.set_ylabel("Rank" if rank else "Cost")
    g.set_xlabel("Optimization Budget")

    path = os.path.join(OPT_PLOTS_DIR, f"{exp + '_rank' if method == 'rank' else exp}.png")
    plt.legend(title="Optimizer")
    plt.tight_layout()
    plt.savefig(path, dpi=500)
    plt.close()


def plot_envs_opt_over_time(algorithm: str, envs: list[str], category_name: str):
    all_opt_data = []
    for env in envs:
        exp = f"{algorithm}_{env}"
        opt_data = read_opt_data(exp)
        if opt_data is None:
            continue
        incumbent_opt_data = get_incumbent(opt_data, exp, "rank")

        incumbent_opt_data["env"] = env
        all_opt_data += [incumbent_opt_data]

    if len (all_opt_data) == 0:
        return

    all_opt_data = pd.concat(all_opt_data)
    
    fig = plt.figure(figsize=(7, 5))
    g = sns.lineplot(data=incumbent_opt_data, x="cum_budget", y="rank", hue="optimizer", errorbar=("ci", 95), hue_order=HUE_ORDER)
    g.set_title(f"{algorithm.upper()} on {category_name}")
    g.set_ylabel("Rank")
    g.set_xlabel("Optimization Budget")

    path = os.path.join(OPT_PLOTS_DIR, f"{algorithm}_{category_name}.png")
    plt.legend(title="Optimizer")
    plt.tight_layout()
    plt.savefig(path, dpi=500)
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

    fig.legend(loc='upper center', bbox_to_anchor=(0.5, 0.1), ncol=4, fancybox=False, shadow=False, frameon=False)

    plt.tight_layout(pad=1.3)
    path = os.path.join(OPT_PLOTS_DIR, f"three_opt.png")
    plt.savefig(path, dpi=500)
    plt.close()


if __name__ == "__main__":
    # plot_3_opt_over_time(["ppo_Acrobot-v1", "dqn_CartPole-v1", "sac_Pendulum-v1"])
    for exp in os.listdir("results/smac_mf"):
       plot_opt_over_time(exp, rank=False)
       plot_opt_over_time(exp, rank=True)
       gc.collect()

    for algorithm, subset_envs in SUBSETS.items():
        plot_envs_opt_over_time(algorithm, subset_envs, "Subset")

    for algorithm, category in ENV_CATEGORIES.items():
        envs = [env for cat_envs in category.values() for env in cat_envs]
        plot_envs_opt_over_time(algorithm, envs, "Full Set")

    for algorithm, category in ENV_CATEGORIES.items():
        for category_name, envs in category.items():
            plot_envs_opt_over_time(algorithm, envs, category_name)