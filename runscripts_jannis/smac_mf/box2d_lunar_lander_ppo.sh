#!/bin/bash


#SBATCH --cpus-per-task=64
#SBATCH --mem=128GB
#SBATCH -J smac_mf_box2d_lunar_lander_ppo
#SBATCH -A hpc-prf-intexml                              # TODO check for your project
#SBATCH -t 7-00:00:00                                   # TODO check for your clusters time limit
#SBATCH --mail-type fail
#SBATCH --mail-user becktepe@stud.uni-hannover.de       # TODO enter your mail and hope slurm isn't reaching out to you :D
#SBATCH -p normal                                       # TODO check for your clusters partition
#SBATCH --output smac_mf/log/smac_mf_box2d_lunar_lander_ppo_%A_%a.out
#SBATCH --error smac_mf/log/smac_mf_box2d_lunar_lander_ppo_%A_%a.err
#SBATCH --array=3-4

cd ..
python runscripts/run_arlbench.py -m --config-name=tune_smac_mf experiments=box2d_lunar_lander_ppo cluster=pc2_cpu smac_seed=$SLURM_ARRAY_TASK_ID 
