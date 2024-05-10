#!/bin/bash
./rs_luis_local.sh cc_acrobot_dqn
./rs_luis_local.sh cc_cartpole_dqn
./rs_luis_local.sh cc_mountain_dqn

./rs_luis_local.sh cc_acrobot_ppo
./rs_luis_local.sh cc_cartpole_ppo
./rs_luis_local.sh cc_continuous_mountain_car_ppo
./rs_luis_local.sh cc_mountain_car_ppo
./rs_luis_local.sh cc_pendulum_ppo

./rs_luis_local.sh cc_continuous_mountain_car_sac
./rs_luis_local.sh cc_pendulum_sac
