import os
import pandas as pd
import logging
import shutil


POPULATION_SIZE = 32
NUM_CONFIGS = 10    # number of dynamic configurations for each run


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
                #continue
        
            all_data = []
            for seed_dir in os.listdir(agent_env_path):
                seed_path = os.path.join(agent_env_path, seed_dir)
                if os.path.isdir(seed_path):
                    csv_file = os.path.join(seed_path, "42", "runhistory.csv")
                    if os.path.isfile(csv_file):
                        try:
                            data = pd.read_csv(csv_file)
                        except pd.errors.ParserError as e:
                            print(f"Error while reading: {csv_file}")
                            print(e)
                            continue
                        data["seed"] = seed_dir

                        if approach == "pbt":
                            # In some cases the hypersweeper started another run that is out-of-budget
                            data = data[data["run_id"] < POPULATION_SIZE * NUM_CONFIGS]

                            # Every 32th entry belongs to the same population member
                            for _, row in data.iterrows():
                                member_id = row["run_id"] % POPULATION_SIZE
                                iteration = row["run_id"] // POPULATION_SIZE
                                data.at[row["run_id"], "member_id"] = member_id
                                data.at[row["run_id"], "iteration"] = iteration
                                data.at[row["run_id"], "budget"] = row["budget"] * (iteration + 1)
                        elif approach == "smac_mf":
                            max_budget = data["budget"].max()
                            max_budget_idx = data[data["budget"] == max_budget].index.max()
                            data = data.iloc[:max_budget_idx + 1, :]
                        all_data.append(data)

                    # remove directories > 255 because we had to many random runs
                    # for some environments due to a bug in the hypersweeper
                    for run_dir in os.listdir(seed_path):
                        run_path = os.path.join(seed_path, run_dir)
                        if not os.path.isdir(run_path):
                            continue
            
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