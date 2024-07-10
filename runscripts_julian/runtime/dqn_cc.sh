#!/bin/bash

#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=2000M                                      
#SBATCH --job-name=rt_cartpole
#SBATCH --account=p0021208
#SBATCH --gres=gpu:1
#SBATCH --partition=c23g
#SBATCH -t 24:00:00            
#SBATCH --mail-type fail                                
#SBATCH --mail-user dierkes@aim.rwth-aachen.de
#SBATCH --output runtime/log/arlb_rs_dqn_cc_%A_%a.out
#SBATCH --error runtime/log/arlb_rs_dqn_cc_%A_%a.err
#SBATCH --array 0-9

cd ..
source /rwthfs/rz/cluster/home/oh751555/i14/arlbench/.venv/bin/activate 
python runscripts/run_runtime_experiments.py algorithm_framework=arlbench seed=$SLURM_ARRAY_TASK_ID +sb_zoo=cc_cartpole_dqn environment.framework=envpool

