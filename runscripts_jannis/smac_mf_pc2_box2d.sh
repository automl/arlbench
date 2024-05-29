#!/bin/bash
./smac_mf_pc2_local.sh box2d_bipedal_walker_sac

./smac_mf_pc2_local.sh box2d_lunar_lander_continuous_ppo
./smac_mf_pc2_local.sh box2d_lunar_lander_continuous_sac

./smac_mf_pc2_local.sh box2d_lunar_lander_dqn
./smac_mf_pc2_local.sh box2d_lunar_lander_ppo
