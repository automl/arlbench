#!/bin/bash
./smac_mf_pc2.sh minigrid_door_key_dqn pc2_cpu
./smac_mf_pc2.sh minigrid_empty_random_dqn pc2_cpu
./smac_mf_pc2.sh minigrid_four_rooms_dqn pc2_cpu
./smac_mf_pc2.sh minigrid_unlock_dqn pc2_cpu

# ./smac_mf_pc2.sh minigrid_door_key_ppo pc2_cpu
# ./smac_mf_pc2.sh minigrid_empty_random_ppo pc2_cpu
# ./smac_mf_pc2.sh minigrid_four_rooms_ppo pc2_cpu
# ./smac_mf_pc2.sh minigrid_unlock_ppo pc2_cpu
