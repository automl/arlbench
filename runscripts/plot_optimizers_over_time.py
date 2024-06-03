import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import os
from scipy.interpolate import interp1d
import numpy as np


OPTIMIZERS = ["rs", "smac", "smac_mf", "pbt"]
OPT_PLOTS_DIR = "plots/optimizer_runs"

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
        all_data += [opt_data]

    return pd.concat(all_data) if len(all_data) > 0 else None


def interpolate(opt_data: pd.DataFrame) -> pd.DataFrame:
    grouped_opt_data = opt_data.groupby(["optimizer", "seed"])

    interpolated_opt_datas = []

    for group_name, group_opt_data in grouped_opt_data:
        optimizer, seed = group_name
        
        sorted_group_opt_data = group_opt_data.sort_values(by="budget")
        
        budgets = sorted_group_opt_data["budget"].values
        performances = sorted_group_opt_data["performance"].values
        
        f = interp1d(budgets, performances, kind="linear")
        
        min_budget, max_budget = np.min(budgets), np.max(budgets)
        if min_budget == max_budget:
            interpolated_opt_data = pd.DataFrame({
                "optimizer": [optimizer] * 100,
                "seed": [seed] * 100,
                "budget": np.linspace(0, opt_data["budget"].max(), 100),
                "performance": performances[0]
            })
        else:
            interpolated_budgets = np.linspace(min_budget, max_budget, 100)
        
            interpolated_performances = f(interpolated_budgets)
            
            interpolated_opt_data = pd.DataFrame({
                "optimizer": [optimizer] * 100,
                "seed": [seed] * 100,
                "budget": interpolated_budgets,
                "performance": interpolated_performances
            })
        
        interpolated_opt_datas.append(interpolated_opt_data)

    return pd.concat(interpolated_opt_datas, ignore_index=True)


def plot_opt_over_time(exp: str):
    opt_data = read_opt_data(exp)
    if opt_data is None:
        return
        
    best_config_opt_data = opt_data.groupby(["optimizer", "budget", "seed"]).apply(lambda x: x.loc[x["performance"].idxmin()]).reset_index(drop=True)


    interpolated_opt_data = interpolate(best_config_opt_data)

    incumbent_opt_data = interpolated_opt_data.sort_values(by=['optimizer', 'seed', 'budget'])
    incumbent_opt_data['incumbent'] = incumbent_opt_data.groupby(['optimizer', 'seed'])['performance'].cummin()
    
    fig = plt.figure(figsize=(5, 5))
    g = sns.lineplot(data=incumbent_opt_data, x="budget", y="incumbent", hue="optimizer", errorbar="sd")
    g.set_ylabel("Cost")
    g.set_xlabel("Budget")

    path = os.path.join(OPT_PLOTS_DIR, f"{exp}.png")
    plt.tight_layout()
    plt.savefig(path, dpi=500)


if __name__ == "__main__":
    for exp in os.listdir("results/pbt"):
        plot_opt_over_time(exp)
