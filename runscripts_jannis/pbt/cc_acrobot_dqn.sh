#!/bin/bash


#SBATCH --cpus-per-task=64
#SBATCH --mem=128GB
#SBATCH -J arlb_pbt_cc_acrobot_dqn
#SBATCH -A hpc-prf-intexml                              # TODO check for your project
#SBATCH -t 7-00:00:00                                   # TODO check for your clusters time limit
#SBATCH --mail-type fail
#SBATCH --mail-user becktepe@stud.uni-hannover.de       # TODO enter your mail and hope slurm isn't reaching out to you :D
#SBATCH -p normal                                       # TODO check for your clusters partition
#SBATCH --output pbt/log/arlb_pbt_cc_acrobot_dqn_%A.out
#SBATCH --error pbt/log/arlb_pbt_cc_acrobot_dqn_%A.err


cd ..
python runscripts/run_arlbench.py -m --config-name=tune_pbt experiments=cc_acrobot_dqn_pbt cluster=local

