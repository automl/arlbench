#!/bin/bash
./smac_mf_pc2.sh box2d_bipedal_walker_sac pc2_cpu

./smac_mf_pc2.sh box2d_lunar_lander_continuous_ppo pc2_cpu
./smac_mf_pc2.sh box2d_lunar_lander_continuous_sac pc2_cpu

# ./smac_mf_pc2_local.sh box2d_lunar_lander_dqn
./smac_mf_pc2.sh box2d_lunar_lander_ppo pc2_cpu
