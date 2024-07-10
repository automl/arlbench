#!/bin/bash
./smac_mf_claix.sh brax_ant_ppo claix_gpu_h100
./smac_mf_claix.sh brax_hopper_ppo claix_gpu_h100
./smac_mf_claix.sh brax_halfcheetah_ppo claix_gpu_h100
./smac_mf_claix.sh brax_humanoid_ppo claix_gpu_h100

#./smac_mf_claix.sh brax_ant_sac claix_gpu_h100
#./smac_mf_claix.sh brax_hopper_sac claix_gpu_h100
#./smac_mf_claix.sh brax_halfcheetah_sac claix_gpu_h100
#./smac_mf_claix.sh brax_humanoid_sac claix_gpu_h100
