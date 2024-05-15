#!/bin/bash


#SBATCH --cpus-per-task=32
#SBATCH --mem=64GB
#SBATCH -J arlb_rs_cc_acrobot_dqn
#SBATCH -t 4-00:00:00                                   # TODO check for your clusters time limit
#SBATCH --mail-type fail
#SBATCH --mail-user becktepe@stud.uni-hannover.de       # TODO enter your mail and hope slurm isn't reaching out to you :D
#SBATCH -p ai                                           # TODO check for your clusters partition
#SBATCH --output sobol/log/arlb_rs_cc_acrobot_dqn_%A_%a.out
#SBATCH --error sobol/log/arlb_rs_cc_acrobot_dqn_%A_%a.err
#SBATCH --array=0-0

cd ..
python runscripts/run_arlbench.py -m --config-name=random_runs autorl.seed=$SLURM_ARRAY_TASK_ID +experiments=cc_acrobot_dqn cluster=local 

