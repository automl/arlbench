#!/bin/bash

#SBATCH --cpus-per-task=4 
#SBATCH --mem-per-cpu=2000M                                      
#SBATCH --job-name=rs_qbert
###SBATCH --account=
#SBATCH -t 128:00:00            
#SBATCH --mail-type fail                                
#SBATCH --mail-user dierkes@aim.rwth-aachen.de
#SBATCH --output sobol/log/arlb_rs_ppo_atari_%A_%a.out
#SBATCH --error sobol/log/arlb_rs_ppo_atari_%A_%a.err
#SBATCH --array 0-9%3

cd ..
source /rwthfs/rz/cluster/home/oh751555/i14/arlbench/.venv/bin/activate
python runscripts/run_arlbench.py -m --config-name=random_runs autorl.seed=$SLURM_ARRAY_TASK_ID algorithm=ppo search_space=ppo_atari environment=atari_qbert cluster=claix_gpu_h100 hydra.sweeper.n_trials=128 hydra.sweeper.sweeper_kwargs.max_parallelization=1.0 hydra.sweeper.sweeper_kwargs.job_array_size_limit=127 hydra.launcher.array_parallelism=128 +sb_zoo=atari_qbert_ppo 

