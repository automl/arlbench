python runscripts/run_arlbench.py -m --config-name=random_runs "cluster=pc2_cpu" "environment=cc_cartpole" "algorithm=dqn" "search_space=dqn" "search_space.seed=0" "autorl.seed=0"
python runscripts/run_arlbench.py -m --config-name=tune_smac "cluster=pc2_cpu" "environment=cc_cartpole" "algorithm=dqn" "search_space=dqn" "smac_seed=0" "autorl.seed=0"
python runscripts/run_arlbench.py -m --config-name=tune_pbt "cluster=pc2_cpu" "environment=cc_cartpole" "algorithm=dqn" "search_space=dqn" "pbt_seed=0" "autorl.seed=0"