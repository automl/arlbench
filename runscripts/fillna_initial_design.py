import pandas as pd
import os


PATH = "runscripts/configs/initial_design"

for file in os.listdir(PATH):
    file_path = os.path.join(PATH, file)

    df = pd.read_csv(file_path)

    # For the initial design in SMAC, we need to replace all NaNs
    # Therefore, we replace them by the worst performance we recorded
    # As SMAC minimizes the cost, the worst performance has the highest value
    if df["performance"].isna().any():
        print(f"Removing NaNs in {file}")
        worst_performance = df["performance"].max()
        df["performance"] = df["performance"].fillna(worst_performance)

        df.to_csv(file_path, index=False)
