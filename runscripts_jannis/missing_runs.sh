#!/bin/bash
echo $1
cd ..
python runscripts/run_arlbench.py --multirun --config-name=random_runs "experiments=$1" "autorl.seed=7"
