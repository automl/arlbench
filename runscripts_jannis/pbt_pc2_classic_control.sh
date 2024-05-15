#!/bin/bash
./pbt_pc2_local.sh cc_acrobot_dqn
./pbt_pc2_local.sh cc_cartpole_dqn
./pbt_pc2_local.sh cc_mountain_car_dqn

./pbt_pc2_local.sh cc_acrobot_ppo
./pbt_pc2_local.sh cc_cartpole_ppo
./pbt_pc2_local.sh cc_continuous_mountain_car_ppo
./pbt_pc2_local.sh cc_mountain_car_ppo
./pbt_pc2_local.sh cc_pendulum_ppo

./pbt_pc2_local.sh cc_continuous_mountain_car_sac
./pbt_pc2_local.sh cc_pendulum_sac
