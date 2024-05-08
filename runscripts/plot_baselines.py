import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd


path = "results/smac/0/dqn_CartPole-v1/0/runhistory.csv"

df = pd.read_csv(path)

sns.lineplot(x="run_id", y="performance", data=df)
plt.savefig(f'plots/smac.png')