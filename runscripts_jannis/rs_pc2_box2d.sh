#!/bin/bash
./rs_pc2.sh box2d_bipedal_walker_ppo pc2_gpu
./rs_pc2.sh box2d_bipedal_walker_sac pc2_gpu

# ./rs_pc2_local_gpu.sh box2d_lunar_lander_continuous_ppo
# ./rs_pc2_local_gpu.sh box2d_lunar_lander_continuous_sac

# ./rs_pc2_local.sh box2d_lunar_lander_dqn
# ./rs_pc2_local.sh box2d_lunar_lander_ppo