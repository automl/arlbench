from __future__ import annotations

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

path = "results/smac/0/dqn_CartPole-v1/0/runhistory.csv"

df = pd.read_csv(path)

sns.lineplot(x="run_id", y="performance", data=df)
plt.savefig("plots/smac.png")
