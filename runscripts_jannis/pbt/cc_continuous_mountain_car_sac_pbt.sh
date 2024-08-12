#!/bin/bash


#SBATCH --cpus-per-task=64
#SBATCH --mem=128GB
#SBATCH -J pbt_cc_continuous_mountain_car_sac_pbt
#SBATCH -A hpc-prf-intexml                              # TODO check for your project
#SBATCH -t 7-00:00:00                                   # TODO check for your clusters time limit
#SBATCH --mail-type fail
#SBATCH --mail-user becktepe@stud.uni-hannover.de       # TODO enter your mail and hope slurm isn't reaching out to you :D
#SBATCH -p normal                                       # TODO check for your clusters partition
#SBATCH --output pbt/log/pbt_cc_continuous_mountain_car_sac_pbt_%A_%a.out
#SBATCH --error pbt/log/pbt_cc_continuous_mountain_car_sac_pbt_%A_%a.err
#SBATCH --array=3-4


cd ..
python runscripts/run_arlbench.py -m --config-name=tune_pbt experiments=cc_continuous_mountain_car_sac_pbt cluster=local pbt_seed=$SLURM_ARRAY_TASK_ID search_space.seed=$SLURM_ARRAY_TASK_ID 

