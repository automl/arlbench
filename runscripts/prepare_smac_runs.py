import ast
import os
import pandas as pd


SMAC_RESULTS = "results/smac"
INCUMBENTS = "runscripts/configs/smac_incumbents"


if __name__ == "__main__":
    for folder in os.listdir(SMAC_RESULTS):
        directory = os.path.join(SMAC_RESULTS, folder)
        if not os.path.isdir(directory):
            continue

        runhistory_path = os.path.join(directory, "42", "runhistory.csv")
        if not os.path.isfile(runhistory_path):
            continue

        runhistory = pd.read_csv(runhistory_path)

    with open('your_file.txt', 'r') as file:
        last_line = file.readlines()[-1]

    last_row_dict = ast.literal_eval(last_line)

    print(last_row_dict)