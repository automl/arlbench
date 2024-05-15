#!/bin/bash
./rs_pc2_local.sh minigrid_door_key_dqn
./rs_pc2_local.sh minigrid_empty_random_dqn
./rs_pc2_local.sh minigrid_four_rooms_dqn
./rs_pc2_local.sh minigrid_unlock_dqn

./rs_pc2_local.sh minigrid_door_key_ppo
./rs_pc2_local.sh minigrid_empty_random_ppo
./rs_pc2_local.sh minigrid_four_rooms_ppo
./rs_pc2_local.sh minigrid_unlock_ppo
