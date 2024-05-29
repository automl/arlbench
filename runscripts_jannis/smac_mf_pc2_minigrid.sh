#!/bin/bash
./smac_mf_pc2_local.sh minigrid_door_key_dqn
./smac_mf_pc2_local.sh minigrid_empty_random_dqn
./smac_mf_pc2_local.sh minigrid_four_rooms_dqn
./smac_mf_pc2_local.sh minigrid_unlock_dqn

./smac_mf_pc2_local.sh minigrid_door_key_ppo
./smac_mf_pc2_local.sh minigrid_empty_random_ppo
./smac_mf_pc2_local.sh minigrid_four_rooms_ppo
./smac_mf_pc2_local.sh minigrid_unlock_ppo
