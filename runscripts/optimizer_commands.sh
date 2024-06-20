#!/bin/bash

# -----------------------------------------------------------------------------------------
# Atari
# -----------------------------------------------------------------------------------------
# DQN
python runscripts/run_arlbench.py -m --config-name=tune_rs "search_space.seed=42,43,44" "experiments=atari_battle_zone_dqn_pbt" 
python runscripts/run_arlbench.py -m --config-name=tune_rs "search_space.seed=42,43,44" "experiments=atari_double_dunk_dqn_pbt"
python runscripts/run_arlbench.py -m --config-name=tune_rs "search_space.seed=42,43,44" "experiments=atari_this_game_dqn_pbt" 
python runscripts/run_arlbench.py -m --config-name=tune_rs "search_space.seed=42,43,44" "experiments=atari_phoenix_dqn_pbt" 
python runscripts/run_arlbench.py -m --config-name=tune_rs "search_space.seed=42,43,44" "experiments=atari_qbert_dqn_pbt" 

python runscripts/run_arlbench.py -m --config-name=tune_smac "smac_seed=0,1,2" "experiments=atari_battle_zone_dqn_pbt" 
python runscripts/run_arlbench.py -m --config-name=tune_smac "smac_seed=0,1,2" "experiments=atari_double_dunk_dqn_pbt"
python runscripts/run_arlbench.py -m --config-name=tune_smac "smac_seed=0,1,2" "experiments=atari_this_game_dqn_pbt" 
python runscripts/run_arlbench.py -m --config-name=tune_smac "smac_seed=0,1,2" "experiments=atari_phoenix_dqn_pbt" 
python runscripts/run_arlbench.py -m --config-name=tune_smac "smac_seed=0,1,2" "experiments=atari_qbert_dqn_pbt" 

python runscripts/run_arlbench.py -m --config-name=tune_smac_mf "smac_seed=0,1,2" "experiments=atari_battle_zone_dqn_pbt" 
python runscripts/run_arlbench.py -m --config-name=tune_smac_mf "smac_seed=0,1,2" "experiments=atari_double_dunk_dqn_pbt"
python runscripts/run_arlbench.py -m --config-name=tune_smac_mf "smac_seed=0,1,2" "experiments=atari_this_game_dqn_pbt" 
python runscripts/run_arlbench.py -m --config-name=tune_smac_mf "smac_seed=0,1,2" "experiments=atari_phoenix_dqn_pbt" 
python runscripts/run_arlbench.py -m --config-name=tune_smac_mf "smac_seed=0,1,2" "experiments=atari_qbert_dqn_pbt" 

python runscripts/run_arlbench.py -m --config-name=tune_pbt "pbt_seed=0,1,2" "experiments=atari_battle_zone_dqn_pbt" 
python runscripts/run_arlbench.py -m --config-name=tune_pbt "pbt_seed=0,1,2" "experiments=atari_double_dunk_dqn_pbt"
python runscripts/run_arlbench.py -m --config-name=tune_pbt "pbt_seed=0,1,2" "experiments=atari_this_game_dqn_pbt" 
python runscripts/run_arlbench.py -m --config-name=tune_pbt "pbt_seed=0,1,2" "experiments=atari_phoenix_dqn_pbt" 
python runscripts/run_arlbench.py -m --config-name=tune_pbt "pbt_seed=0,1,2" "experiments=atari_qbert_dqn_pbt" 

# PPO
python runscripts/run_arlbench.py -m --config-name=tune_rs "search_space.seed=42,43,44" "experiments=atari_battle_zone_ppo" 
python runscripts/run_arlbench.py -m --config-name=tune_rs "search_space.seed=42,43,44" "experiments=atari_double_dunk_ppo"
python runscripts/run_arlbench.py -m --config-name=tune_rs "search_space.seed=42,43,44" "experiments=atari_this_game_ppo" 
python runscripts/run_arlbench.py -m --config-name=tune_rs "search_space.seed=42,43,44" "experiments=atari_phoenix_ppo" 
python runscripts/run_arlbench.py -m --config-name=tune_rs "search_space.seed=42,43,44" "experiments=atari_qbert_ppo" 

python runscripts/run_arlbench.py -m --config-name=tune_smac "smac_seed=0,1,2" "experiments=atari_battle_zone_ppo" 
python runscripts/run_arlbench.py -m --config-name=tune_smac "smac_seed=0,1,2" "experiments=atari_double_dunk_ppo"
python runscripts/run_arlbench.py -m --config-name=tune_smac "smac_seed=0,1,2" "experiments=atari_this_game_ppo" 
python runscripts/run_arlbench.py -m --config-name=tune_smac "smac_seed=0,1,2" "experiments=atari_phoenix_ppo" 
python runscripts/run_arlbench.py -m --config-name=tune_smac "smac_seed=0,1,2" "experiments=atari_qbert_ppo" 

python runscripts/run_arlbench.py -m --config-name=tune_smac_mf "smac_seed=0,1,2" "experiments=atari_battle_zone_ppo" 
python runscripts/run_arlbench.py -m --config-name=tune_smac_mf "smac_seed=0,1,2" "experiments=atari_double_dunk_ppo"
python runscripts/run_arlbench.py -m --config-name=tune_smac_mf "smac_seed=0,1,2" "experiments=atari_this_game_ppo" 
python runscripts/run_arlbench.py -m --config-name=tune_smac_mf "smac_seed=0,1,2" "experiments=atari_phoenix_ppo" 
python runscripts/run_arlbench.py -m --config-name=tune_smac_mf "smac_seed=0,1,2" "experiments=atari_qbert_ppo" 

python runscripts/run_arlbench.py -m --config-name=tune_pbt "pbt_seed=0,1,2" "experiments=atari_battle_zone_ppo" 
python runscripts/run_arlbench.py -m --config-name=tune_pbt "pbt_seed=0,1,2" "experiments=atari_double_dunk_ppo"
python runscripts/run_arlbench.py -m --config-name=tune_pbt "pbt_seed=0,1,2" "experiments=atari_this_game_ppo" 
python runscripts/run_arlbench.py -m --config-name=tune_pbt "pbt_seed=0,1,2" "experiments=atari_phoenix_ppo" 
python runscripts/run_arlbench.py -m --config-name=tune_pbt "pbt_seed=0,1,2" "experiments=atari_qbert_ppo" 

# -----------------------------------------------------------------------------------------
# Box2D
# -----------------------------------------------------------------------------------------
# DQN
python runscripts/run_arlbench.py -m --config-name=tune_rs "search_space.seed=42,43,44" "experiments=box2d_lunar_lander_dqn_pbt" 
python runscripts/run_arlbench.py -m --config-name=tune_smac "smac_seed=0,1,2" "experiments=box2d_lunar_lander_dqn_pbt" 
python runscripts/run_arlbench.py -m --config-name=tune_smac_mf "smac_seed=0,1,2" "experiments=box2d_lunar_lander_dqn_pbt" 
python runscripts/run_arlbench.py -m --config-name=tune_pbt "pbt_seed=0,1,2" "experiments=box2d_lunar_lander_dqn_pbt" 

# PPO
python runscripts/run_arlbench.py -m --config-name=tune_rs "search_space.seed=42,43,44" "experiments=box2d_lunar_lander_ppo" 
python runscripts/run_arlbench.py -m --config-name=tune_smac "smac_seed=0,1,2" "experiments=box2d_lunar_lander_ppo" 
python runscripts/run_arlbench.py -m --config-name=tune_smac_mf "smac_seed=0,1,2" "experiments=box2d_lunar_lander_ppo" 
python runscripts/run_arlbench.py -m --config-name=tune_pbt "pbt_seed=0,1,2" "experiments=box2d_lunar_lander_ppo" 

python runscripts/run_arlbench.py -m --config-name=tune_rs "search_space.seed=42,43,44" "experiments=box2d_lunar_lander_continuous_ppo" 
python runscripts/run_arlbench.py -m --config-name=tune_smac "smac_seed=0,1,2" "experiments=box2d_lunar_lander_continuous_ppo" 
python runscripts/run_arlbench.py -m --config-name=tune_smac_mf "smac_seed=0,1,2" "experiments=box2d_lunar_lander_continuous_ppo" 
python runscripts/run_arlbench.py -m --config-name=tune_pbt "pbt_seed=0,1,2" "experiments=box2d_lunar_lander_continuous_ppo" 

# SAC
python runscripts/run_arlbench.py -m --config-name=tune_rs "search_space.seed=42,43,44" "experiments=box2d_bipedal_walker_sac_pbt" 
python runscripts/run_arlbench.py -m --config-name=tune_smac "smac_seed=0,1,2" "experiments=box2d_bipedal_walker_sac_pbt" 
python runscripts/run_arlbench.py -m --config-name=tune_smac_mf "smac_seed=0,1,2" "experiments=box2d_bipedal_walker_sac_pbt" 
python runscripts/run_arlbench.py -m --config-name=tune_pbt "pbt_seed=0,1,2" "experiments=box2d_bipedal_walker_sac_pbt" 

python runscripts/run_arlbench.py -m --config-name=tune_rs "search_space.seed=42,43,44" "experiments=box2d_lunar_lander_continuous_sac_pbt" 
python runscripts/run_arlbench.py -m --config-name=tune_smac "smac_seed=0,1,2" "experiments=box2d_lunar_lander_continuous_sac_pbt" 
python runscripts/run_arlbench.py -m --config-name=tune_smac_mf "smac_seed=0,1,2" "experiments=box2d_lunar_lander_continuous_sac_pbt" 
python runscripts/run_arlbench.py -m --config-name=tune_pbt "pbt_seed=0,1,2" "experiments=box2d_lunar_lander_continuous_sac_pbt" 

# -----------------------------------------------------------------------------------------
# Brax
# -----------------------------------------------------------------------------------------
# PPO
python runscripts/run_arlbench.py -m --config-name=tune_rs "search_space.seed=42,43,44" "experiments=brax_ant_ppo" 
python runscripts/run_arlbench.py -m --config-name=tune_rs "search_space.seed=42,43,44" "experiments=brax_halfcheetah_ppo"
python runscripts/run_arlbench.py -m --config-name=tune_rs "search_space.seed=42,43,44" "experiments=brax_hopper_ppo" 
python runscripts/run_arlbench.py -m --config-name=tune_rs "search_space.seed=42,43,44" "experiments=brax_humanoid_ppo" 

python runscripts/run_arlbench.py -m --config-name=tune_smac "smac_seed=0,1,2" "experiments=brax_ant_ppo" 
python runscripts/run_arlbench.py -m --config-name=tune_smac "smac_seed=0,1,2" "experiments=brax_halfcheetah_ppo"
python runscripts/run_arlbench.py -m --config-name=tune_smac "smac_seed=0,1,2" "experiments=brax_hopper_ppo" 
python runscripts/run_arlbench.py -m --config-name=tune_smac "smac_seed=0,1,2" "experiments=brax_humanoid_ppo" 

python runscripts/run_arlbench.py -m --config-name=tune_smac_mf "smac_seed=0,1,2" "experiments=brax_ant_ppo" 
python runscripts/run_arlbench.py -m --config-name=tune_smac_mf "smac_seed=0,1,2" "experiments=brax_halfcheetah_ppo"
python runscripts/run_arlbench.py -m --config-name=tune_smac_mf "smac_seed=0,1,2" "experiments=brax_hopper_ppo" 
python runscripts/run_arlbench.py -m --config-name=tune_smac_mf "smac_seed=0,1,2" "experiments=brax_humanoid_ppo" 

python runscripts/run_arlbench.py -m --config-name=tune_pbt "pbt_seed=0,1,2" "experiments=brax_ant_ppo" 
python runscripts/run_arlbench.py -m --config-name=tune_pbt "pbt_seed=0,1,2" "experiments=brax_halfcheetah_ppo"
python runscripts/run_arlbench.py -m --config-name=tune_pbt "pbt_seed=0,1,2" "experiments=brax_hopper_ppo" 
python runscripts/run_arlbench.py -m --config-name=tune_pbt "pbt_seed=0,1,2" "experiments=brax_humanoid_ppo" 

# SAC
python runscripts/run_arlbench.py -m --config-name=tune_rs "search_space.seed=42,43,44" "experiments=brax_ant_sac_pbt" 
python runscripts/run_arlbench.py -m --config-name=tune_rs "search_space.seed=42,43,44" "experiments=brax_halfcheetah_sac_pbt"
python runscripts/run_arlbench.py -m --config-name=tune_rs "search_space.seed=42,43,44" "experiments=brax_hopper_sac_pbt" 
python runscripts/run_arlbench.py -m --config-name=tune_rs "search_space.seed=42,43,44" "experiments=brax_humanoid_sac_pbt" 

python runscripts/run_arlbench.py -m --config-name=tune_smac "smac_seed=0,1,2" "experiments=brax_ant_sac_pbt" 
python runscripts/run_arlbench.py -m --config-name=tune_smac "smac_seed=0,1,2" "experiments=brax_halfcheetah_sac_pbt"
python runscripts/run_arlbench.py -m --config-name=tune_smac "smac_seed=0,1,2" "experiments=brax_hopper_sac_pbt" 
python runscripts/run_arlbench.py -m --config-name=tune_smac "smac_seed=0,1,2" "experiments=brax_humanoid_sac_pbt" 

python runscripts/run_arlbench.py -m --config-name=tune_smac_mf "smac_seed=0,1,2" "experiments=brax_ant_sac_pbt" 
python runscripts/run_arlbench.py -m --config-name=tune_smac_mf "smac_seed=0,1,2" "experiments=brax_halfcheetah_sac_pbt"
python runscripts/run_arlbench.py -m --config-name=tune_smac_mf "smac_seed=0,1,2" "experiments=brax_hopper_sac_pbt" 
python runscripts/run_arlbench.py -m --config-name=tune_smac_mf "smac_seed=0,1,2" "experiments=brax_humanoid_sac_pbt" 

python runscripts/run_arlbench.py -m --config-name=tune_pbt "pbt_seed=0,1,2" "experiments=brax_ant_sac_pbt" 
python runscripts/run_arlbench.py -m --config-name=tune_pbt "pbt_seed=0,1,2" "experiments=brax_halfcheetah_sac_pbt"
python runscripts/run_arlbench.py -m --config-name=tune_pbt "pbt_seed=0,1,2" "experiments=brax_hopper_sac_pbt" 
python runscripts/run_arlbench.py -m --config-name=tune_pbt "pbt_seed=0,1,2" "experiments=brax_humanoid_sac_pbt" 

# -----------------------------------------------------------------------------------------
# Classic Control
# -----------------------------------------------------------------------------------------
# DQN
python runscripts/run_arlbench.py -m --config-name=tune_rs "search_space.seed=42,43,44" "experiments=cc_acrobot_dqn_pbt" 
python runscripts/run_arlbench.py -m --config-name=tune_rs "search_space.seed=42,43,44" "experiments=cc_cartpole_dqn_pbt"
python runscripts/run_arlbench.py -m --config-name=tune_rs "search_space.seed=42,43,44" "experiments=cc_mountain_car_dqn_pbt" 

python runscripts/run_arlbench.py -m --config-name=tune_smac "smac_seed=0,1,2" "experiments=cc_acrobot_dqn_pbt" 
python runscripts/run_arlbench.py -m --config-name=tune_smac "smac_seed=0,1,2" "experiments=cc_cartpole_dqn_pbt"
python runscripts/run_arlbench.py -m --config-name=tune_smac "smac_seed=0,1,2" "experiments=cc_mountain_car_dqn_pbt" 

python runscripts/run_arlbench.py -m --config-name=tune_smac_mf "smac_seed=0,1,2" "experiments=cc_acrobot_dqn_pbt" 
python runscripts/run_arlbench.py -m --config-name=tune_smac_mf "smac_seed=0,1,2" "experiments=cc_cartpole_dqn_pbt"
python runscripts/run_arlbench.py -m --config-name=tune_smac_mf "smac_seed=0,1,2" "experiments=cc_mountain_car_dqn_pbt" 

python runscripts/run_arlbench.py -m --config-name=tune_pbt "pbt_seed=0,1,2" "experiments=cc_acrobot_dqn_pbt" 
python runscripts/run_arlbench.py -m --config-name=tune_pbt "pbt_seed=0,1,2" "experiments=cc_cartpole_dqn_pbt"
python runscripts/run_arlbench.py -m --config-name=tune_pbt "pbt_seed=0,1,2" "experiments=cc_mountain_car_dqn_pbt" 

# PPO
python runscripts/run_arlbench.py -m --config-name=tune_rs "search_space.seed=42,43,44" "experiments=cc_acrobot_ppo" 
python runscripts/run_arlbench.py -m --config-name=tune_rs "search_space.seed=42,43,44" "experiments=cc_cartpole_ppo"
python runscripts/run_arlbench.py -m --config-name=tune_rs "search_space.seed=42,43,44" "experiments=cc_mountain_car_ppo" 
python runscripts/run_arlbench.py -m --config-name=tune_rs "search_space.seed=42,43,44" "experiments=cc_continuous_mountain_car_ppo" 
python runscripts/run_arlbench.py -m --config-name=tune_rs "search_space.seed=42,43,44" "experiments=cc_pendulum_ppo" 

python runscripts/run_arlbench.py -m --config-name=tune_smac "smac_seed=0,1,2" "experiments=cc_acrobot_ppo" 
python runscripts/run_arlbench.py -m --config-name=tune_smac "smac_seed=0,1,2" "experiments=cc_cartpole_ppo"
python runscripts/run_arlbench.py -m --config-name=tune_smac "smac_seed=0,1,2" "experiments=cc_mountain_car_ppo" 
python runscripts/run_arlbench.py -m --config-name=tune_smac "smac_seed=0,1,2" "experiments=cc_continuous_mountain_car_ppo" 
python runscripts/run_arlbench.py -m --config-name=tune_smac "smac_seed=0,1,2" "experiments=cc_pendulum_ppo" 

python runscripts/run_arlbench.py -m --config-name=tune_smac_mf "smac_seed=0,1,2" "experiments=cc_acrobot_ppo" 
python runscripts/run_arlbench.py -m --config-name=tune_smac_mf "smac_seed=0,1,2" "experiments=cc_cartpole_ppo"
python runscripts/run_arlbench.py -m --config-name=tune_smac_mf "smac_seed=0,1,2" "experiments=cc_mountain_car_ppo" 
python runscripts/run_arlbench.py -m --config-name=tune_smac_mf "smac_seed=0,1,2" "experiments=cc_continuous_mountain_car_ppo" 
python runscripts/run_arlbench.py -m --config-name=tune_smac_mf "smac_seed=0,1,2" "experiments=cc_pendulum_ppo" 

python runscripts/run_arlbench.py -m --config-name=tune_pbt "pbt_seed=0,1,2" "experiments=cc_acrobot_ppo" 
python runscripts/run_arlbench.py -m --config-name=tune_pbt "pbt_seed=0,1,2" "experiments=cc_cartpole_ppo"
python runscripts/run_arlbench.py -m --config-name=tune_pbt "pbt_seed=0,1,2" "experiments=cc_mountain_car_ppo" 
python runscripts/run_arlbench.py -m --config-name=tune_pbt "pbt_seed=0,1,2" "experiments=cc_continuous_mountain_car_ppo" 
python runscripts/run_arlbench.py -m --config-name=tune_pbt "pbt_seed=0,1,2" "experiments=cc_pendulum_ppo" 

# SAC
python runscripts/run_arlbench.py -m --config-name=tune_rs "search_space.seed=42,43,44" "experiments=cc_continuous_mountain_car_sac_pbt" 
python runscripts/run_arlbench.py -m --config-name=tune_rs "search_space.seed=42,43,44" "experiments=cc_pendulum_sac_pbt"

python runscripts/run_arlbench.py -m --config-name=tune_smac "smac_seed=0,1,2" "experiments=cc_continuous_mountain_car_sac_pbt" 
python runscripts/run_arlbench.py -m --config-name=tune_smac "smac_seed=0,1,2" "experiments=cc_pendulum_sac_pbt"

python runscripts/run_arlbench.py -m --config-name=tune_smac_mf "smac_seed=0,1,2" "experiments=cc_continuous_mountain_car_sac_pbt" 
python runscripts/run_arlbench.py -m --config-name=tune_smac_mf "smac_seed=0,1,2" "experiments=cc_pendulum_sac_pbt" 

python runscripts/run_arlbench.py -m --config-name=tune_pbt "pbt_seed=0,1,2" "experiments=cc_continuous_mountain_car_sac_pbt" 
python runscripts/run_arlbench.py -m --config-name=tune_pbt "pbt_seed=0,1,2" "experiments=cc_pendulum_sac_pbt"

# -----------------------------------------------------------------------------------------
# XLand
# -----------------------------------------------------------------------------------------
# DQN
python runscripts/run_arlbench.py -m --config-name=tune_rs "search_space.seed=42,43,44" "experiments=minigrid_door_key_dqn_pbt" 
python runscripts/run_arlbench.py -m --config-name=tune_rs "search_space.seed=42,43,44" "experiments=minigrid_empty_random_dqn_pbt"
python runscripts/run_arlbench.py -m --config-name=tune_rs "search_space.seed=42,43,44" "experiments=minigrid_four_rooms_dqn_pbt" 
python runscripts/run_arlbench.py -m --config-name=tune_rs "search_space.seed=42,43,44" "experiments=minigrid_unlock_dqn_pbt" 

python runscripts/run_arlbench.py -m --config-name=tune_smac "smac_seed=0,1,2" "experiments=minigrid_door_key_dqn_pbt" 
python runscripts/run_arlbench.py -m --config-name=tune_smac "smac_seed=0,1,2" "experiments=minigrid_empty_random_dqn_pbt"
python runscripts/run_arlbench.py -m --config-name=tune_smac "smac_seed=0,1,2" "experiments=minigrid_four_rooms_dqn_pbt" 
python runscripts/run_arlbench.py -m --config-name=tune_smac "smac_seed=0,1,2" "experiments=minigrid_unlock_dqn_pbt" 

python runscripts/run_arlbench.py -m --config-name=tune_smac_mf "smac_seed=0,1,2" "experiments=minigrid_door_key_dqn_pbt" 
python runscripts/run_arlbench.py -m --config-name=tune_smac_mf "smac_seed=0,1,2" "experiments=minigrid_empty_random_dqn_pbt"
python runscripts/run_arlbench.py -m --config-name=tune_smac_mf "smac_seed=0,1,2" "experiments=minigrid_four_rooms_dqn_pbt" 
python runscripts/run_arlbench.py -m --config-name=tune_smac_mf "smac_seed=0,1,2" "experiments=minigrid_unlock_dqn_pbt" 

python runscripts/run_arlbench.py -m --config-name=tune_pbt "pbt_seed=0,1,2" "experiments=minigrid_door_key_dqn_pbt" 
python runscripts/run_arlbench.py -m --config-name=tune_pbt "pbt_seed=0,1,2" "experiments=minigrid_empty_random_dqn_pbt"
python runscripts/run_arlbench.py -m --config-name=tune_pbt "pbt_seed=0,1,2" "experiments=minigrid_four_rooms_dqn_pbt" 
python runscripts/run_arlbench.py -m --config-name=tune_pbt "pbt_seed=0,1,2" "experiments=minigrid_unlock_dqn_pbt" 

# PPO
python runscripts/run_arlbench.py -m --config-name=tune_rs "search_space.seed=42,43,44" "experiments=minigrid_door_key_ppo" 
python runscripts/run_arlbench.py -m --config-name=tune_rs "search_space.seed=42,43,44" "experiments=minigrid_empty_random_ppo"
python runscripts/run_arlbench.py -m --config-name=tune_rs "search_space.seed=42,43,44" "experiments=minigrid_four_rooms_ppo" 
python runscripts/run_arlbench.py -m --config-name=tune_rs "search_space.seed=42,43,44" "experiments=minigrid_unlock_ppo" 

python runscripts/run_arlbench.py -m --config-name=tune_smac "smac_seed=0,1,2" "experiments=minigrid_door_key_ppo" 
python runscripts/run_arlbench.py -m --config-name=tune_smac "smac_seed=0,1,2" "experiments=minigrid_empty_random_ppo"
python runscripts/run_arlbench.py -m --config-name=tune_smac "smac_seed=0,1,2" "experiments=minigrid_four_rooms_ppo" 
python runscripts/run_arlbench.py -m --config-name=tune_smac "smac_seed=0,1,2" "experiments=minigrid_unlock_ppo" 

python runscripts/run_arlbench.py -m --config-name=tune_smac_mf "smac_seed=0,1,2" "experiments=minigrid_door_key_ppo" 
python runscripts/run_arlbench.py -m --config-name=tune_smac_mf "smac_seed=0,1,2" "experiments=minigrid_empty_random_ppo"
python runscripts/run_arlbench.py -m --config-name=tune_smac_mf "smac_seed=0,1,2" "experiments=minigrid_four_rooms_ppo" 
python runscripts/run_arlbench.py -m --config-name=tune_smac_mf "smac_seed=0,1,2" "experiments=minigrid_unlock_ppo" 

python runscripts/run_arlbench.py -m --config-name=tune_pbt "pbt_seed=0,1,2" "experiments=minigrid_door_key_ppo" 
python runscripts/run_arlbench.py -m --config-name=tune_pbt "pbt_seed=0,1,2" "experiments=minigrid_empty_random_ppo"
python runscripts/run_arlbench.py -m --config-name=tune_pbt "pbt_seed=0,1,2" "experiments=minigrid_four_rooms_ppo" 
python runscripts/run_arlbench.py -m --config-name=tune_pbt "pbt_seed=0,1,2" "experiments=minigrid_unlock_ppo" 


