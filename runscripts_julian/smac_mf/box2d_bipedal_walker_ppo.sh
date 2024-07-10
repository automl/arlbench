#!/bin/bash


#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=2000M
#SBATCH --job-name=smac_box2d_bipedal_walker_ppo
#SBATCH --account=p0021208
#SBATCH --time 48:00:00                                 
#SBATCH --mail-type fail
#SBATCH --mail-user dierkes.julian@rwth-aachen.de
#SBATCH --output smac_mf/log/smac_box2d_bipedal_walker_ppo_%A.out
#SBATCH --error smac_mf/log/smac_box2d_bipedal_walker_ppo_%A.err
#SBATCH --array 0,1,3

cd ..
source /rwthfs/rz/cluster/home/oh751555/i14/arlbench/.venv/bin/activate
python runscripts/run_arlbench.py -m --config-name=tune_smac_mf experiments=box2d_bipedal_walker_ppo cluster=claix_cpu smac_seed=$SLURM_ARRAY_TASK_ID

