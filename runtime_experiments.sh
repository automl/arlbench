# Classic Control
python runscripts/run_experiments.py -m "seed=range(10)" "cluster=pc2_gpu" "algorithm_framework=arlbench" "+sb_zoo=cc_cartpole_ppo,cc_cartpole_dqn" "n_eval_steps=1,100" 
python runscripts/run_experiments.py -m "seed=range(10)" "cluster=pc2_gpu" "algorithm_framework=purejaxrl" "+sb_zoo=cc_cartpole_ppo,cc_cartpole_dqn" 
python runscripts/run_experiments.py -m "seed=range(10)" "cluster=pc2_gpu" "algorithm_framework=arlbench" "+sb_zoo=glob(cc_*)" "environment.framework=envpool" "n_eval_steps=1,100" 
python runscripts/run_experiments.py -m "seed=range(10)" "cluster=pc2_gpu" "algorithm_framework=sbx" "+sb_zoo=glob(cc_*)" "environment.framework=envpool" "n_eval_steps=1,100"

# Mujoco Walkers
python runscripts/run_experiments.py -m "seed=range(10)" "cluster=pc2_gpu" "algorithm_framework=arlbench,sbx" "+sb_zoo=glob(mujoco_*)" "n_eval_steps=1,100"

# Atari
python runscripts/run_experiments.py -m "seed=range(10)" "cluster=pc2_gpu" "algorithm_framework=arlbench,sbx" "+sb_zoo=glob(atari_*)" "n_eval_steps=1,100"

# Procgen
python runscripts/run_experiments.py -m "seed=range(10)" "cluster=pc2_gpu" "algorithm_framework=arlbench" "+sb_zoo=procgen_bigfish_easy_ppo,procgen_bigfish_easy_dqn,procgen_bigfish_hard_ppo,procgen_bigfish_hard_dqn" "n_eval_steps=1,100"
