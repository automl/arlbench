#!/bin/bash

#SBATCH --cpus-per-task=4 
#SBATCH --mem-per-cpu=2000M                                      
#SBATCH --job-name=rs_atari
###SBATCH --account=
#SBATCH -t 128:00:00                                   # TODO check for your clusters time limit
#SBATCH --mail-type fail                                
#SBATCH --mail-user becktepe@stud.uni-hannover.de       # TODO enter your mail and hope slurm isn't reaching out to you :D
#SBATCH --output sobol/log/arlb_rs___%A_%a.out
#SBATCH --error sobol/log/arlb_rs___%A_%a.err
#SBATCH --array=0-9

cd ..
source /rwthfs/rz/cluster/home/oh751555/i14/arlbench/.venv/bin/activate
python runscripts/run_arlbench.py -m --config-name=random_runs autorl.seed=$SLURM_ARRAY_TASK_ID algorithm= search_space= environment= cluster=

