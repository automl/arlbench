python runscripts/run_runtime_experiments.py -m "seed=range(20)" "cluster=pc2_cpu" "algorithm_framework=arlbench" "+sb_zoo=cc_cartpole_ppo,cc_cartpole_dqn" "n_total_timesteps=1000000"
python runscripts/run_runtime_experiments.py -m "seed=range(20)" "cluster=pc2_cpu" "algorithm_framework=purejaxrl" "+sb_zoo=cc_cartpole_ppo,cc_cartpole_dqn" "n_total_timesteps=1000000"

python runscripts/run_runtime_experiments.py -m "seed=range(20)" "cluster=pc2_gpu" "algorithm_framework=arlbench" "+sb_zoo=cc_cartpole_ppo,cc_cartpole_dqn" "n_total_timesteps=1000000"
python runscripts/run_runtime_experiments.py -m "seed=range(20)" "cluster=pc2_gpu" "algorithm_framework=purejaxrl" "+sb_zoo=cc_cartpole_ppo,cc_cartpole_dqn" "n_total_timesteps=1000000"