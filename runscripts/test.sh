# Classic Control
python runscripts/run_experiments.py -m algorithm_framework=arlbench,purejaxrl +sb_zoo=cc_cartpole_ppo
python runscripts/run_experiments.py -m algorithm_framework=arlbench,purejaxrl +sb_zoo=cc_cartpole_dqn
python runscripts/run_experiments.py -m algorithm_framework=arlbench,purejaxrl +sb_zoo=cc_pendulum_sac
python runscripts/run_experiments.py -m algorithm_framework=sbx +sb_zoo=cc_pendulum_sac environment.framework=envpool
python runscripts/run_experiments.py -m algorithm_framework=sbx +sb_zoo=cc_pendulum_sac environment.framework=envpool

# Mujoco Walkers
python runscripts/run_experiments.py -m algorithm_framework=arlbench,sbx +sb_zoo=mujoco_ant_ppo environment.n_totaltimesteps=1e5
python runscripts/run_experiments.py -m algorithm_framework=arlbench,sbx +sb_zoo=mujoco_ant_sac environment.n_totaltimesteps=1e5
python runscripts/run_experiments.py -m algorithm_framework=arlbench,sbx +sb_zoo=mujoco_halfcheetah_ppo environment.n_totaltimesteps=1e5
python runscripts/run_experiments.py -m algorithm_framework=arlbench,sbx +sb_zoo=mujoco_halfcheetah_sac environment.n_totaltimesteps=1e5

# Atari
python runscripts/run_experiments.py -m algorithm_framework=arlbench,sbx +sb_zoo=atari_pong_ppo environment.n_totaltimesteps=1e5
python runscripts/run_experiments.py -m algorithm_framework=arlbench,sbx +sb_zoo=atari_pong_dqn environment.n_totaltimesteps=1e5

# Procgen
python runscripts/run_experiments.py -m algorithm_framework=arlbench,sbx +sb_zoo=procgen_bigfish_easy_ppo environment.n_totaltimesteps=1e5
