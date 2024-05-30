#!/bin/bash


#SBATCH --cpus-per-task=32
#SBATCH --mem=64GB
#SBATCH -J rs_cc_mountain_car_dqn
#SBATCH -A hpc-prf-intexml                              # TODO check for your project
#SBATCH -t 7-00:00:00                                   # TODO check for your clusters time limit
#SBATCH --mail-type fail
#SBATCH --mail-user becktepe@stud.uni-hannover.de       # TODO enter your mail and hope slurm isn't reaching out to you :D
#SBATCH -p normal                                       # TODO check for your clusters partition
#SBATCH --output rs/log/rs_cc_mountain_car_dqn_%A_%a.out
#SBATCH --error rs/log/rs_cc_mountain_car_dqn_%A_%a.err
#SBATCH --array=43-44

cd ..
python runscripts/run_arlbench.py -m --config-name=tune_rs experiments=cc_mountain_car_dqn cluster=local search_space.seed=$SLURM_ARRAY_TASK_ID 

