#!/bin/bash

#SBATCH --cpus-per-task=4 
#SBATCH --mem-per-cpu=2000M                                      
#SBATCH --job-name=rs_atari
###SBATCH --account=
#SBATCH -t 128:00:00            
#SBATCH --mail-type fail                                
#SBATCH --mail-user dierkes@aim.rwth-aachen.de
#SBATCH --output sobol/log/arlb_rs_ppo_atari_breakout_%A_%a.out
#SBATCH --error sobol/log/arlb_rs_ppo_atari_breakout_%A_%a.err
#SBATCH --array 0-10%2

cd ..
source /rwthfs/rz/cluster/home/oh751555/i14/arlbench/.venv/bin/activate
python runscripts/run_arlbench.py -m --config-name=random_runs hydra.launcher.array_parallelism=256 autorl.seed=$SLURM_ARRAY_TASK_ID algorithm=ppo search_space=ppo_8env environment=atari_breakout cluster=claix_gpu_h100 +experiments=atari_breakout_ppo 

