#!/bin/bash
./pbt_pc2.sh minigrid_door_key_dqn_pbt pc2_cpu
./pbt_pc2.sh minigrid_empty_random_dqn_pbt pc2_cpu
./pbt_pc2.sh minigrid_four_rooms_dqn_pbt pc2_cpu
./pbt_pc2.sh minigrid_unlock_dqn_pbt pc2_cpu

./pbt_pc2.sh minigrid_door_key_ppo pc2_cpu
./pbt_pc2.sh minigrid_empty_random_ppo pc2_cpu
./pbt_pc2.sh minigrid_four_rooms_ppo pc2_cpu
./pbt_pc2.sh minigrid_unlock_ppo pc2_cpu
