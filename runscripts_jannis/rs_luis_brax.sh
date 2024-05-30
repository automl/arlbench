#!/bin/bash
./rs_luis_local_gpu.sh brax_ant_ppo 
./rs_luis_local_gpu.sh brax_halfcheetah_ppo 
./rs_luis_local_gpu.sh brax_hopper_ppo 
./rs_luis_local_gpu.sh brax_humanoid_ppo 

./rs_luis_local_gpu.sh brax_ant_sac
./rs_luis_local_gpu.sh brax_halfcheetah_sac 
./rs_luis_local_gpu.sh brax_hopper_sac 
./rs_luis_local_gpu.sh brax_humanoid_sac 

