#!/bin/bash
./smac_mf_pc2_local.sh cc_acrobot_dqn
./smac_mf_pc2_local.sh cc_cartpole_dqn
./smac_mf_pc2_local.sh cc_mountain_car_dqn

./smac_mf_pc2_local.sh cc_acrobot_ppo
./smac_mf_pc2_local.sh cc_cartpole_ppo
./smac_mf_pc2_local.sh cc_continuous_mountain_car_ppo
./smac_mf_pc2_local.sh cc_mountain_car_ppo
./smac_mf_pc2_local.sh cc_pendulum_ppo

./smac_mf_pc2_local.sh cc_continuous_mountain_car_sac
./smac_mf_pc2_local.sh cc_pendulum_sac
