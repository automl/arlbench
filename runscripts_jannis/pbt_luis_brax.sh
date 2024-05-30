#!/bin/bash
./pbt_luis_local_gpu.sh brax_ant_ppo 
./pbt_luis_local_gpu.sh brax_halfcheetah_ppo 
./pbt_luis_local_gpu.sh brax_hopper_ppo 
./pbt_luis_local_gpu.sh brax_humanoid_ppo 

./pbt_luis_local_gpu.sh brax_ant_sac_pbt
./pbt_luis_local_gpu.sh brax_halfcheetah_sac_pbt
./pbt_luis_local_gpu.sh brax_hopper_sac_pbt
./pbt_luis_local_gpu.sh brax_humanoid_sac_pbt

