#!/bin/bash
./rs_pc2_local_gpu.sh brax_ant_ppo 
./rs_pc2_local_gpu.sh brax_halfcheetah_ppo 
./rs_pc2_local_gpu.sh brax_hopper_ppo 
./rs_pc2_local_gpu.sh brax_humanoid_ppo 

./rs_pc2_local_gpu.sh brax_ant_sac 
./rs_pc2_local_gpu.sh brax_halfcheetah_sac 
./rs_pc2_local_gpu.sh brax_hopper_sac 
./rs_pc2_local_gpu.sh brax_humanoid_sac 
