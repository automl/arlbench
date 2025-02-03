import os
import pandas as pd
import numpy as np
import math
import logging
import shutil


def aggregate_runhistories(approach: str):
    base_path = os.path.join("results", approach)

    if not os.path.exists(base_path):
        logging.info(f"Directory for {approach} does not exist.")
        return

    for algorithm_env in os.listdir(base_path):
        agent_env_path = os.path.join(base_path, algorithm_env)
        if os.path.isdir(agent_env_path):
            combined_path = os.path.join(agent_env_path, "runhistory_combined.csv")

            if os.path.exists(combined_path):
                logging.info(f"Aggregated runhistory already exists for {algorithm_env}. Skipping...")
                continue

            all_data = []
            for seed_dir in os.listdir(agent_env_path):
                seed_path = os.path.join(agent_env_path, seed_dir)
                csv_file = os.path.join(seed_path, "runhistory.csv")
                if os.path.isfile(csv_file):
                    try:
                        data = pd.read_csv(csv_file)
                    except pd.errors.ParserError as e:
                        print(f"Error while reading: {csv_file}")
                        print(e)
                        continue
                    data["seed"] = seed_dir
                    dynamic_data = pd.DataFrame()
                for train_dir in range(1024):
                    train_path = os.path.join(seed_path, str(train_dir))
                    if os.path.isdir(train_path):
                        csv_file = os.path.join(train_path, "evaluation.csv")
                        if os.path.isfile(csv_file):
                            try:
                                train_steps_data = pd.read_csv(csv_file)
                            except pd.errors.ParserError as e:
                                print(f"Error while reading: {csv_file}")
                                print(e)
                                continue
                    cur_row = data.iloc[int(train_dir)].copy()
                    math.isclose(
                        np.abs(float(cur_row["performance"])),
                        np.abs(float(train_steps_data["returns"].iloc[-1])),
                        rel_tol=0.01
                    )

                    assert len(train_steps_data) == 10
                    cur_row = pd.DataFrame([cur_row] * len(train_steps_data))
                    train_steps_data = pd.concat(
                        [cur_row.reset_index(drop=True), train_steps_data.reset_index(drop=True)], axis=1
                    )
                    train_steps_data["performance"] = train_steps_data["returns"]
                    train_steps_data["budget"] = train_steps_data["steps"]
                    train_steps_data = train_steps_data.drop("returns", axis=1)
                    train_steps_data = train_steps_data.drop("steps", axis=1)
                    dynamic_data = pd.concat(
                        [dynamic_data, train_steps_data], axis=0
                    )

                dynamic_data = dynamic_data.reset_index(drop=True)
                all_data.append(dynamic_data)

                # remove directories > 255 because we had to many random runs
                # for some environments due to a bug in the hypersweeper
                for run_dir in os.listdir(seed_path):
                    run_path = os.path.join(seed_path, run_dir)
                    if not os.path.isdir(run_path):
                        continue

                    # We do not need this directory
                    if run_dir == "checkpoints":
                        print(f"Removing {run_path}")
                        shutil.rmtree(run_path)
                        continue

                    try:
                        run_dir_num = int(run_dir)
                        if run_dir_num > 1023:
                            print(f"Removing {run_path}")
                            shutil.rmtree(run_path)
                    except:
                        pass

            if all_data:
                combined_data = pd.concat(all_data, ignore_index=True)
                combined_data.to_csv(combined_path, index=False)
                logging.info(f"Created combined runhistory for {algorithm_env}.")
            else:
                logging.info(f"No data found for {algorithm_env}. Skipping...")


if __name__ == "__main__":
    import sys
    if len(sys.argv) != 2:
        print("Usage: python script_name.py [approach]")
        sys.exit(1)
    approach = sys.argv[1]
    aggregate_runhistories(approach)
