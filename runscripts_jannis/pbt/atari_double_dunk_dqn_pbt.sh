#!/bin/bash


#SBATCH --cpus-per-task=8
#SBATCH --mem=128GB
#SBATCH -J pbt_atari_double_dunk_dqn_pbt
#SBATCH -A hpc-prf-intexml                              # TODO check for your project
#SBATCH -t 7-00:00:00                                   # TODO check for your clusters time limit
#SBATCH --mail-type fail
#SBATCH --mail-user becktepe@stud.uni-hannover.de       # TODO enter your mail and hope slurm isn't reaching out to you :D
#SBATCH -p normal                                       # TODO check for your clusters partition
#SBATCH --output pbt/log/pbt_atari_double_dunk_dqn_pbt_%A_%a.out
#SBATCH --error pbt/log/pbt_atari_double_dunk_dqn_pbt_%A_%a.err
#SBATCH --array=4

cd ..
python runscripts/run_arlbench.py -m --config-name=tune_pbt experiments=atari_double_dunk_dqn_pbt cluster=pc2_gpu pbt_seed=$SLURM_ARRAY_TASK_ID search_space.seed=$SLURM_ARRAY_TASK_ID
