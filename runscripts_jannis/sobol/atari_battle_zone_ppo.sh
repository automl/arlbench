#!/bin/bash


#SBATCH --cpus-per-task=32
#SBATCH --mem=64GB
#SBATCH -J arlb_rs_atari_battle_zone_ppo
#SBATCH -A hpc-prf-intexml                              # TODO check for your project
#SBATCH -t 7-00:00:00                                   # TODO check for your clusters time limit
#SBATCH --mail-type fail
#SBATCH --mail-user becktepe@stud.uni-hannover.de       # TODO enter your mail and hope slurm isn't reaching out to you :D
#SBATCH -p normal                                       # TODO check for your clusters partition
#SBATCH --output sobol/log/arlb_rs_atari_battle_zone_ppo_%A_%a.out
#SBATCH --error sobol/log/arlb_rs_atari_battle_zone_ppo_%A_%a.err
#SBATCH --array=5-5


cd ..
python runscripts/run_arlbench.py -m --config-name=random_runs autorl.seed=$SLURM_ARRAY_TASK_ID experiments=atari_battle_zone_ppo cluster=local 

