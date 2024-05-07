#!/bin/bash


#SBATCH --cpus-per-task=8
#SBATCH --mem=16GB
#SBATCH -J arlb_rs_ppo_atari_battle_zone
#SBATCH -A hpc-prf-intexml                              # TODO check for your project
#SBATCH -t 7-00:00:00                                   # TODO check for your clusters time limit
#SBATCH --mail-type fail
#SBATCH --mail-user becktepe@stud.uni-hannover.de       # TODO enter your mail and hope slurm isn't reaching out to you :D
#SBATCH -p normal                                       # TODO check for your clusters partition
#SBATCH --output sobol/log/arlb_rs_ppo_atari_battle_zone_%A_%a.out
#SBATCH --error sobol/log/arlb_rs_ppo_atari_battle_zone_%A_%a.err
#SBATCH --array=0-9

cd ..
python runscripts/run_arlbench.py -m --config-name=random_runs autorl.seed=$SLURM_ARRAY_TASK_ID algorithm=ppo search_space=ppo environment=atari_battle_zone cluster=pc2_gpu
