from __future__ import annotations

import os

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def plot_compare_trainings(df_1, df_1_name, df_2, df_2_name, plot_name, save_dir):
    df_1["type"] = df_1_name
    df_2["type"] = df_2_name
    df = pd.concat([df_1, df_2])
    ax = sns.lineplot(data=df, x="step", y="return", hue="type", errorbar="sd")
    ax.set_title(plot_name)
    plt.legend(loc="upper left")
    plt.savefig(os.path.join(save_dir, f"{plot_name}.png"))


def get_mean_df(path):
    df = pd.DataFrame()
    for i in range(1, 4):
        cur_df = pd.read_csv(os.path.join(path, f"{i}_results.csv"))
        mean_return = cur_df.mean(axis=0)
        mean_return = mean_return.transpose()
        mean_return = mean_return.reset_index()
        mean_return.columns = ["step", "return"]
        mean_return["step"] = range(len(mean_return))
        df = pd.concat([df, mean_return])
    return df


if __name__ == "__main__":
    df_1 = get_mean_df(os.path.join("./dqn_results", "envpool_Pong-v5", "8env_gpu"))
    df_2 = get_mean_df(os.path.join("./dqn_results", "envpool_Pong-v5", "4env_gpu"))
    plot_compare_trainings(
        df_1, "ARLB_8env", df_2, "ARLB_4env", "DQN CartPol", "dqn_results"
    )
