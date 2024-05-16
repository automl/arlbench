import os
import pandas as pd
import logging


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
                if os.path.isdir(seed_path):
                    csv_file = os.path.join(seed_path, "runhistory.csv")
                    print(csv_file)
                    if os.path.isfile(csv_file):
                        data = pd.read_csv(csv_file)
                        data["seed"] = seed_dir
                        all_data.append(data)
            
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