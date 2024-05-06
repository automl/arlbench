#!/bin/bash
./run_rs.sh dqn atari_battle_zone pc2_gpu
./run_rs.sh dqn atari_double_dunk pc2_gpu

./run_rs.sh ppo atari_battle_zone pc2_gpu
./run_rs.sh ppo atari_double_dunk pc2_gpu
