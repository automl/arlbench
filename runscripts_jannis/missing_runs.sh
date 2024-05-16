#!/bin/bash
echo $1
cd ..
for i in 0 1 2 3 4 5 6 8 9 
do
    python runscripts/run_arlbench.py --multirun --config-name=random_runs "experiments=$1" "autorl.seed=$i"
done