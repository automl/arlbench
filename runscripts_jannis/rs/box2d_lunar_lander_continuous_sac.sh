#!/bin/bash


#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=64GB
#SBATCH -J rs_box2d_lunar_lander_continuous_sac
#SBATCH -A hpc-prf-intexml                              # TODO check for your project
#SBATCH -t 2-00:00:00                                   # TODO check for your clusters time limit
#SBATCH --mail-type fail
#SBATCH --mail-user becktepe@stud.uni-hannover.de       # TODO enter your mail and hope slurm isn't reaching out to you :D
#SBATCH -p gpu                                       # TODO check for your clusters partition
#SBATCH --output rs/log/rs_box2d_lunar_lander_continuous_sac_%A_%a.out
#SBATCH --error rs/log/rs_box2d_lunar_lander_continuous_sac_%A_%a.err
#SBATCH --array=43-44

cd ..
python runscripts/run_arlbench.py -m --config-name=tune_rs experiments=box2d_lunar_lander_continuous_sac cluster=local search_space.seed=$SLURM_ARRAY_TASK_ID 

