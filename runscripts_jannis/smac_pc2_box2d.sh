#!/bin/bash
# ./rs_pc2.sh box2d_bipedal_walker_ppo pc2_cpu
./smac_pc2_local_gpu.sh box2d_bipedal_walker_sac

./smac_pc2_local_gpu.sh box2d_lunar_lander_continuous_ppo
./smac_pc2_local_gpu.sh box2d_lunar_lander_continuous_sac

# ./smac_pc2_local.sh box2d_lunar_lander_dqn
# ./smac_pc2_local.sh box2d_lunar_lander_ppo
