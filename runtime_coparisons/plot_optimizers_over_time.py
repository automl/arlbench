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
    result_path = os.path.join(OPT_RESULTS_DIR, f"{exp}.csv")
    if os.path.isfile(result_path):
        return pd.read_csv(result_path)

    interpolated_opt_datas = []

    min_budget = opt_data[opt_data["optimizer"] == "rs"]["cum_budget"].min()
    max_budget = opt_data[opt_data["optimizer"] == "rs"]["cum_budget"].max()

    for (optimizer, seed), group_opt_data in opt_data.groupby(["optimizer", "seed"]):
        max_opt_budget = group_opt_data["cum_budget"].max()

        new_cum_budget = pd.Series(np.arange(int(min_budget), int(max_opt_budget) + 1, 100), name="cum_budget")
        df_merged = pd.DataFrame(new_cum_budget)

        df_merged = pd.merge(df_merged, group_opt_data, on="cum_budget", how="left")
        interpolated_opt_data = df_merged.interpolate(method="linear")
        interpolated_opt_data[interpolated_opt_data["cum_budget"] <= max_budget]

        interpolated_opt_data["seed"] = seed
        interpolated_opt_data["optimizer"] = OPTIMIZER_NAMES[optimizer]
    
        interpolated_opt_datas.append(interpolated_opt_data)

    df = pd.concat(interpolated_opt_datas, ignore_index=True)
    interpolated_data = df[df["cum_budget"] % STEP_SIZE == 0]
    del df

    interpolated_data.to_csv(result_path, index=False)

    return interpolated_data


def plot_opt_over_time(exp: str):
    opt_data = read_opt_data(exp)
    if opt_data is None:
        return

    algorithm, environment = exp.split("_")
    
    # W need to select the best configuration that we have for each budget
    best_config_opt_data = opt_data.groupby(["optimizer", "cum_budget", "seed"]).apply(lambda x: x.loc[x["performance"].idxmin()]).reset_index(drop=True)

    interpolated_opt_data = interpolate(best_config_opt_data)

    incumbent_opt_data = interpolated_opt_data.sort_values(by=["optimizer", "seed", "cum_budget"])
    incumbent_opt_data["incumbent"] = incumbent_opt_data.groupby(["optimizer", "seed"])["performance"].cummin()
    
    fig = plt.figure(figsize=(7, 5))
    g = sns.lineplot(data=incumbent_opt_data, x="cum_budget", y="incumbent", hue="optimizer", errorbar="sd")
    g.set_title(f"{algorithm.upper()} on {environment}")
    g.set_ylabel("Cost")
    g.set_xlabel("Optimization Budget")

    path = os.path.join(OPT_PLOTS_DIR, f"{exp}.png")
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
        axes[i].set_xlabel("Budget")
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
        try:
            plot_opt_over_time(exp)
            gc.collect()
        except:
            pass
