#!/bin/bash
./smac_pc2.sh brax_ant_ppo pc2_gpu
./smac_pc2.sh brax_halfcheetah_ppo pc2_gpu
./smac_pc2.sh brax_hopper_ppo pc2_gpu
./smac_pc2.sh brax_humanoid_ppo pc2_gpu

./smac_pc2.sh brax_ant_sac pc2_gpu
./smac_pc2.sh brax_halfcheetah_sac pc2_gpu
./smac_pc2.sh brax_hopper_sac pc2_gpu
./smac_pc2.sh brax_humanoid_sac pc2_gpu
