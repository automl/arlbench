#!/bin/bash

# -----------------------------------------------------------------------------------------
# Atari
# -----------------------------------------------------------------------------------------
# DQN
python runscripts/run_arlbench.py -m --config-name=random_runs "autorl.seed=range(10)" "experiments=atari_battle_zone_dqn" "hydra.sweeper.n_trials=128"
python runscripts/run_arlbench.py -m --config-name=random_runs "autorl.seed=range(10)" "experiments=atari_double_dunk_dqn" "hydra.sweeper.n_trials=128"
python runscripts/run_arlbench.py -m --config-name=random_runs "autorl.seed=range(10)" "experiments=atari_this_game_dqn" "hydra.sweeper.n_trials=128"
python runscripts/run_arlbench.py -m --config-name=random_runs "autorl.seed=range(10)" "experiments=atari_phoenix_dqn" "hydra.sweeper.n_trials=128"
python runscripts/run_arlbench.py -m --config-name=random_runs "autorl.seed=range(10)" "experiments=atari_qbert_dqn" "hydra.sweeper.n_trials=128"

# PPO
python runscripts/run_arlbench.py -m --config-name=random_runs "autorl.seed=range(10)" "experiments=atari_battle_zone_ppo" "hydra.sweeper.n_trials=128"
python runscripts/run_arlbench.py -m --config-name=random_runs "autorl.seed=range(10)" "experiments=atari_double_dunk_ppo" "hydra.sweeper.n_trials=128"
python runscripts/run_arlbench.py -m --config-name=random_runs "autorl.seed=range(10)" "experiments=atari_this_game_ppo" "hydra.sweeper.n_trials=128"
python runscripts/run_arlbench.py -m --config-name=random_runs "autorl.seed=range(10)" "experiments=atari_phoenix_ppo" "hydra.sweeper.n_trials=128"
python runscripts/run_arlbench.py -m --config-name=random_runs "autorl.seed=range(10)" "experiments=atari_qbert_ppo" "hydra.sweeper.n_trials=128"

# -----------------------------------------------------------------------------------------
# Box2D
# -----------------------------------------------------------------------------------------
# DQN
python runscripts/run_arlbench.py -m --config-name=random_runs "autorl.seed=range(10)" "experiments=box2d_lunar_lander_dqn"

# PPO
python runscripts/run_arlbench.py -m --config-name=random_runs "autorl.seed=range(10)" "experiments=box2d_lunar_lander_ppo"
python runscripts/run_arlbench.py -m --config-name=random_runs "autorl.seed=range(10)" "experiments=box2d_lunar_lander_continuous_ppo"

# SAC
python runscripts/run_arlbench.py -m --config-name=random_runs "autorl.seed=range(10)" "experiments=box2d_bipedal_walker_sac"
python runscripts/run_arlbench.py -m --config-name=random_runs "autorl.seed=range(10)" "experiments=box2d_lunar_lander_continuous_sac"

# -----------------------------------------------------------------------------------------
# Brax
# -----------------------------------------------------------------------------------------
# PPO
python runscripts/run_arlbench.py -m --config-name=random_runs "autorl.seed=range(10)" "experiments=brax_ant_ppo"
python runscripts/run_arlbench.py -m --config-name=random_runs "autorl.seed=range(10)" "experiments=brax_halfcheetah_ppo"
python runscripts/run_arlbench.py -m --config-name=random_runs "autorl.seed=range(10)" "experiments=brax_hopper_ppo"
python runscripts/run_arlbench.py -m --config-name=random_runs "autorl.seed=range(10)" "experiments=brax_humanoid_ppo"

# SAC
python runscripts/run_arlbench.py -m --config-name=random_runs "autorl.seed=range(10)" "experiments=brax_ant_sac"
python runscripts/run_arlbench.py -m --config-name=random_runs "autorl.seed=range(10)" "experiments=brax_halfcheetah_sac"
python runscripts/run_arlbench.py -m --config-name=random_runs "autorl.seed=range(10)" "experiments=brax_hopper_sac"
python runscripts/run_arlbench.py -m --config-name=random_runs "autorl.seed=range(10)" "experiments=brax_humanoid_sac"

# -----------------------------------------------------------------------------------------
# Classic Control
# -----------------------------------------------------------------------------------------
# DQN
python runscripts/run_arlbench.py -m --config-name=random_runs "autorl.seed=range(10)" "experiments=cc_acrobot_dqn"
python runscripts/run_arlbench.py -m --config-name=random_runs "autorl.seed=range(10)" "experiments=cc_cartpole_dqn"
python runscripts/run_arlbench.py -m --config-name=random_runs "autorl.seed=range(10)" "experiments=cc_mountain_car_dqn"

# PPO
python runscripts/run_arlbench.py -m --config-name=random_runs "autorl.seed=range(10)" "experiments=cc_acrobot_ppo"
python runscripts/run_arlbench.py -m --config-name=random_runs "autorl.seed=range(10)" "experiments=cc_cartpole_ppo"
python runscripts/run_arlbench.py -m --config-name=random_runs "autorl.seed=range(10)" "experiments=cc_mountain_car_ppo"
python runscripts/run_arlbench.py -m --config-name=random_runs "autorl.seed=range(10)" "experiments=cc_continuous_mountain_car_ppo"
python runscripts/run_arlbench.py -m --config-name=random_runs "autorl.seed=range(10)" "experiments=cc_pendulum_ppo"

# SAC
python runscripts/run_arlbench.py -m --config-name=random_runs "autorl.seed=range(10)" "experiments=cc_continuous_mountain_car_sac"
python runscripts/run_arlbench.py -m --config-name=random_runs "autorl.seed=range(10)" "experiments=cc_pendulum_sac"

# -----------------------------------------------------------------------------------------
# XLand
# -----------------------------------------------------------------------------------------
# DQN
python runscripts/run_arlbench.py -m --config-name=random_runs "autorl.seed=range(10)" "experiments=minigrid_door_key_dqn"
python runscripts/run_arlbench.py -m --config-name=random_runs "autorl.seed=range(10)" "experiments=minigrid_empty_random_dqn"
python runscripts/run_arlbench.py -m --config-name=random_runs "autorl.seed=range(10)" "experiments=minigrid_four_rooms_dqn"
python runscripts/run_arlbench.py -m --config-name=random_runs "autorl.seed=range(10)" "experiments=minigrid_unlock_dqn"

# PPO
python runscripts/run_arlbench.py -m --config-name=random_runs "autorl.seed=range(10)" "experiments=minigrid_door_key_ppo"
python runscripts/run_arlbench.py -m --config-name=random_runs "autorl.seed=range(10)" "experiments=minigrid_empty_random_ppo"
python runscripts/run_arlbench.py -m --config-name=random_runs "autorl.seed=range(10)" "experiments=minigrid_four_rooms_ppo"
python runscripts/run_arlbench.py -m --config-name=random_runs "autorl.seed=range(10)" "experiments=minigrid_unlock_ppo"

