import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import os
import sys

proc_gen_envs = [
    "BigfishEasy-v0",
    "CoinrunEasy-v0",
    "FruitbotEasy-v0",
    "JumperEasy-v0",
    "LeaperEasy-v0",
    "MazeEasy-v0",
    "MinerEasy-v0",
    "StarpilotEasy-v0",
    "NinjaEasy-v0",
    "PlunderEasy-v0",
    "BossfightEasy-v0",
    "ClimberEasy-v0",
    "DodgeballEasy-v0",
    "CaveflyerEasy-v0",
    "HeistEasy-v0",
    "ChaserEasy-v0",
]

for env in proc_gen_envs:
#env = sys.argv[1]
    plot_dir = "procgen_baselines"
    df = pd.DataFrame()
    for seed in range(9):
        path = os.path.join("..", "results", f"ppo_{env}", str(seed), "0", "evaluation.csv")
        data = pd.read_csv(path)
        data["seed"] = seed
        df = pd.concat([df, data])

    sns.lineplot(data=df, x="steps", y="returns", errorbar="sd")
    plt.xlabel("Iteration")
    plt.ylabel("Performance")
    plt.title(env)
    os.makedirs(os.path.join("..", "plots", plot_dir), exist_ok=True)
    plt.savefig(os.path.join("..", "plots", plot_dir, f"ppo_{env}.png"))
    plt.close()