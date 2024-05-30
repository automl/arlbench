#!/bin/bash


#SBATCH --cpus-per-task=32
#SBATCH --mem=64GB
#SBATCH -J rs_box2d_lunar_lander_continuous_ppo
#SBATCH -A hpc-prf-intexml                              # TODO check for your project
#SBATCH -t 7-00:00:00                                   # TODO check for your clusters time limit
#SBATCH --mail-type fail
#SBATCH --mail-user becktepe@stud.uni-hannover.de       # TODO enter your mail and hope slurm isn't reaching out to you :D
#SBATCH -p normal                                       # TODO check for your clusters partition
#SBATCH --output rs/log/rs_box2d_lunar_lander_continuous_ppo_%A_%a.out
#SBATCH --error rs/log/rs_box2d_lunar_lander_continuous_ppo_%A_%a.err
#SBATCH --array=43-44

cd ..
python runscripts/run_arlbench.py -m --config-name=tune_rs experiments=box2d_lunar_lander_continuous_ppo cluster=local search_space.seed=$SLURM_ARRAY_TASK_ID 

