import os
import pandas as pd
import logging
import shutil


N_CONFIGS = {
    "rs": 32,
    "smac": 32,
    "smac_mf": 70,
    "pbt": 320
}

N_SEEDS = 5


def aggregate_runhistories(approach: str):
    base_path = os.path.join("results_finished", approach)
    
    if not os.path.exists(base_path):
        logging.info(f"Directory for {approach} does not exist.")
        return

    for algorithm_env in os.listdir(base_path):
        agent_env_path = os.path.join(base_path, algorithm_env)
        if os.path.isdir(agent_env_path):
            combined_runhistory_path = os.path.join(agent_env_path, "runhistory_combined.csv")
            combined_incumbent_path = os.path.join(agent_env_path, "incumbent_combined.csv")

            all_runhistories = []
            all_incumbents = []

            for seed_dir in os.listdir(agent_env_path):
                seed_path = os.path.join(agent_env_path, seed_dir)
                if os.path.isdir(seed_path):
                    runhistory_file = os.path.join(seed_path, "42", "runhistory.csv")
                    incumbent_file = os.path.join(seed_path, "42", "incumbent.csv")
                    if os.path.isfile(runhistory_file) and os.path.isfile(incumbent_file):
                        try:
                            runhistory = pd.read_csv(runhistory_file)
                            incumbent = pd.read_csv(incumbent_file)
                        except pd.errors.ParserError as e:
                            print(f"Error while reading directory {seed_path}")
                            continue
                        runhistory["optimisation_seed"] = int(seed_dir)
                        incumbent["optimisation_seed"] = int(seed_dir)

                        if approach == "pbt" and len(runhistory) == N_CONFIGS[approach] + 1:
                            runhistory = runhistory.iloc[:N_CONFIGS[approach], :]

                        if not len(runhistory) == N_CONFIGS[approach]:
                            print(f"{runhistory_file}: Expected {N_CONFIGS[approach]} entries but got {len(runhistory)} instead.")
                            continue

                        def detangle_seeds(row):
                            trimmed_row = row.copy()
                            trimmed_row = trimmed_row.drop(
                                ["performance_seed_0", "performance_seed_1", "performance_seed_2"],
                            )
                            new_rows = []
                            for i in range(3):
                                n_row = trimmed_row.copy()
                                n_row["performance"] = row[f"performance_seed_{i}"]
                                n_row["seed"] = i
                                new_rows.append(n_row)
                            return new_rows
                        all_new_rows = runhistory.apply(detangle_seeds, axis=1)
                        detangled_rows = []
                        for rows in all_new_rows:
                            detangled_rows += rows
                        detangled_runhistory = pd.DataFrame(detangled_rows)

                        all_runhistories.append(detangled_runhistory)
                        all_incumbents.append(incumbent)

            if all_runhistories:
                all_runhistories = pd.concat(all_runhistories, ignore_index=True)
                all_runhistories.to_csv(combined_runhistory_path, index=False)

                all_incumbents = pd.concat(all_incumbents, ignore_index=True)
                all_incumbents.to_csv(combined_incumbent_path, index=False)

                if not len(all_runhistories) == N_SEEDS * N_CONFIGS[approach] * 3:  # 3 training seeds
                    print(f"{algorithm_env} Runhistory: Expected {N_SEEDS * N_CONFIGS[approach]} entries but got {len(all_runhistories)} instead.")

                if not len(all_incumbents) == N_SEEDS * N_CONFIGS[approach]:
                    print(f"{algorithm_env} Incumbent: Expected {N_SEEDS * N_CONFIGS[approach]} entries but got {len(all_incumbents)} instead.")

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
