#!/bin/bash


#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=2000M
#SBATCH --job-name=rs_box2d_bipedal_walker_ppo
#SBATCH --partition=c23mm
#SBATCH --account=p0021208
#SBATCH -t 12:00:00                                 
#SBATCH --mail-type fail
#SBATCH --mail-user dierkes.julian@rwth-aachen.de
#SBATCH --output rs/log/rs_box2d_bipedal_walker_ppo_%A.out
#SBATCH --error rs/log/rs_box2d_bipedal_walker_ppo_%A.err
#SBATCH --array=42-46

cd ..
source /rwthfs/rz/cluster/home/oh751555/i14/arlbench/.venv/bin/activate
python runscripts/run_arlbench.py -m --config-name=tune_rs experiments=box2d_bipedal_walker_ppo cluster=claix_cpu search_space.seed=$SLURM_ARRAY_TASK_ID

