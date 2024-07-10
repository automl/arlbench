#!/bin/bash
./run_rs_claix.sh ppo atari phoenix claix_gpu_h100 256 256
./run_rs_claix.sh ppo atari qbert claix_gpu_h100 256 256
./run_rs_claix.sh ppo atari this_game claix_gpu_h100 256 256
./run_rs_claix.sh ppo atari battle_zone claix_gpu_h100 256 256
./run_rs_claix.sh ppo atari double_dunk claix_gpu_h100 256 256

./run_rs_claix.sh dqn atari phoenix claix_gpu_h100 256 256
./run_rs_claix.sh dqn atari qbert claix_gpu_h100 256 256
./run_rs_claix.sh dqn atari this_game claix_gpu_h100 256 256
./run_rs_claix.sh dqn atari battle_zone claix_gpu_h100 256 256
./run_rs_claix.sh dqn atari double_dunk claix_gpu_h100 256 256
