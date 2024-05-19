#!/bin/bash

# -----------------------------------------------------------------------------------------
# Atari
# -----------------------------------------------------------------------------------------
# DQN
python runscripts/run_arlbench.py -m --config-name=random_runs "autorl.seed=%SEED%" "experiments=atari_battle_zone_dqn"
python runscripts/run_arlbench.py -m --config-name=random_runs "autorl.seed=%SEED%" "experiments=atari_double_dunk_dqn"
python runscripts/run_arlbench.py -m --config-name=random_runs "autorl.seed=%SEED%" "experiments=atari_phoenix_dqn"
python runscripts/run_arlbench.py -m --config-name=random_runs "autorl.seed=%SEED%" "experiments=atari_this_game_dqn"

# PPO
python runscripts/run_arlbench.py -m --config-name=random_runs "autorl.seed=%SEED%" "experiments=atari_battle_zone_ppo"
python runscripts/run_arlbench.py -m --config-name=random_runs "autorl.seed=%SEED%" "experiments=atari_double_dunk_ppo"
python runscripts/run_arlbench.py -m --config-name=random_runs "autorl.seed=%SEED%" "experiments=atari_phoenix_ppo"
python runscripts/run_arlbench.py -m --config-name=random_runs "autorl.seed=%SEED%" "experiments=atari_this_game_ppo"

# -----------------------------------------------------------------------------------------
# Box2D
# -----------------------------------------------------------------------------------------
# DQN
python runscripts/run_arlbench.py -m --config-name=random_runs "autorl.seed=%SEED%" "experiments=box2d_lunar_lander_dqn"

# PPO
python runscripts/run_arlbench.py -m --config-name=random_runs "autorl.seed=%SEED%" "experiments=box2d_bipedal_walker_ppo"
python runscripts/run_arlbench.py -m --config-name=random_runs "autorl.seed=%SEED%" "experiments=box2d_lunar_lander_ppo"
python runscripts/run_arlbench.py -m --config-name=random_runs "autorl.seed=%SEED%" "experiments=box2d_lunar_lander_continuous_ppo"

# SAC
python runscripts/run_arlbench.py -m --config-name=random_runs "autorl.seed=%SEED%" "experiments=box2d_bipedal_walker_sac"
python runscripts/run_arlbench.py -m --config-name=random_runs "autorl.seed=%SEED%" "experiments=box2d_lunar_lander_continuous_sac"

# -----------------------------------------------------------------------------------------
# Brax
# -----------------------------------------------------------------------------------------
# PPO
python runscripts/run_arlbench.py -m --config-name=random_runs "autorl.seed=%SEED%" "experiments=brax_ant_ppo"
python runscripts/run_arlbench.py -m --config-name=random_runs "autorl.seed=%SEED%" "experiments=brax_halfcheetah_ppo"
python runscripts/run_arlbench.py -m --config-name=random_runs "autorl.seed=%SEED%" "experiments=brax_hopper_ppo"
python runscripts/run_arlbench.py -m --config-name=random_runs "autorl.seed=%SEED%" "experiments=brax_humanoid_ppo"

# SAC
python runscripts/run_arlbench.py -m --config-name=random_runs "autorl.seed=%SEED%" "experiments=brax_ant_sac"
python runscripts/run_arlbench.py -m --config-name=random_runs "autorl.seed=%SEED%" "experiments=brax_halfcheetah_sac"
python runscripts/run_arlbench.py -m --config-name=random_runs "autorl.seed=%SEED%" "experiments=brax_hopper_sac"
python runscripts/run_arlbench.py -m --config-name=random_runs "autorl.seed=%SEED%" "experiments=brax_humanoid_sac"

# -----------------------------------------------------------------------------------------
# Classic Control
# -----------------------------------------------------------------------------------------
# DQN
python runscripts/run_arlbench.py -m --config-name=random_runs "autorl.seed=%SEED%" "experiments=cc_acrobot_dqn"
python runscripts/run_arlbench.py -m --config-name=random_runs "autorl.seed=%SEED%" "experiments=cc_cartpole_dqn"
python runscripts/run_arlbench.py -m --config-name=random_runs "autorl.seed=%SEED%" "experiments=cc_mountain_car_dqn"

# PPO
python runscripts/run_arlbench.py -m --config-name=random_runs "autorl.seed=%SEED%" "experiments=cc_acrobot_ppo"
python runscripts/run_arlbench.py -m --config-name=random_runs "autorl.seed=%SEED%" "experiments=cc_cartpole_ppo"
python runscripts/run_arlbench.py -m --config-name=random_runs "autorl.seed=%SEED%" "experiments=cc_mountain_car_ppo"
python runscripts/run_arlbench.py -m --config-name=random_runs "autorl.seed=%SEED%" "experiments=cc_continuous_mountain_car_ppo"
python runscripts/run_arlbench.py -m --config-name=random_runs "autorl.seed=%SEED%" "experiments=cc_pendulum_ppo"

# SAC
python runscripts/run_arlbench.py -m --config-name=random_runs "autorl.seed=%SEED%" "experiments=cc_continuous_mountain_car_sac"
python runscripts/run_arlbench.py -m --config-name=random_runs "autorl.seed=%SEED%" "experiments=cc_pendulum_sac"

# -----------------------------------------------------------------------------------------
# Minigrid
# -----------------------------------------------------------------------------------------
# DQN
python runscripts/run_arlbench.py -m --config-name=random_runs "autorl.seed=%SEED%" "experiments=minigrid_door_key_dqn"
python runscripts/run_arlbench.py -m --config-name=random_runs "autorl.seed=%SEED%" "experiments=minigrid_empty_random_dqn"
python runscripts/run_arlbench.py -m --config-name=random_runs "autorl.seed=%SEED%" "experiments=minigrid_four_rooms_dqn"
python runscripts/run_arlbench.py -m --config-name=random_runs "autorl.seed=%SEED%" "experiments=minigrid_unlock_dqn"

# PPO
python runscripts/run_arlbench.py -m --config-name=random_runs "autorl.seed=%SEED%" "experiments=minigrid_door_key_ppo"
python runscripts/run_arlbench.py -m --config-name=random_runs "autorl.seed=%SEED%" "experiments=minigrid_empty_random_ppo"
python runscripts/run_arlbench.py -m --config-name=random_runs "autorl.seed=%SEED%" "experiments=minigrid_four_rooms_ppo"
python runscripts/run_arlbench.py -m --config-name=random_runs "autorl.seed=%SEED%" "experiments=minigrid_unlock_ppo"

