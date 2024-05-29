#!/bin/bash
./pbt_pc2_local.sh minigrid_door_key_dqn_pbt
./pbt_pc2_local.sh minigrid_empty_random_dqn_pbt
./pbt_pc2_local.sh minigrid_four_rooms_dqn_pbt
./pbt_pc2_local.sh minigrid_unlock_dqn_pbt

./pbt_pc2_local.sh minigrid_door_key_ppo
./pbt_pc2_local.sh minigrid_empty_random_ppo
./pbt_pc2_local.sh minigrid_four_rooms_ppo
./pbt_pc2_local.sh minigrid_unlock_ppo
