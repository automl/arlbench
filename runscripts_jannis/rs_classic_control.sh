#!/bin/bash
./run_rs.sh dqn cc_acrobot local
./run_rs.sh dqn cc_cartpole local
./run_rs.sh dqn cc_mountain_car local

./run_rs.sh ppo cc_acrobot local
./run_rs.sh ppo cc_cartpole local
./run_rs.sh ppo cc_continuous_mountain_car local
./run_rs.sh ppo cc_mountain_car local
./run_rs.sh ppo cc_pendulum local

./run_rs.sh sac cc_continuous_mountain_car local
./run_rs.sh sac cc_pendulum local
