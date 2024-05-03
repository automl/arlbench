#!/bin/bash
#for configuration options see: https://uni-paderborn.atlassian.net/wiki/spaces/PC2DOK/pages/12944324/Running+Compute+Jobs
#SBATCH -N 1
#SBATCH -n 4                                    # Number of CPUs you want on that machine (<=128)
#SBATCH --gres=gpu:a100:1
#SBATCH --mem-per-cpu=4GB
#SBATCH --time=24:00:00

#SBATCH --job-name=arlb
#SBATCH -A hpc-prf-intexml                      # Project name, do not change

#SBATCH --partition=gpu

#SBATCH --mail-user=becktepe@stud.uni-hannover.de
#SBATCH --mail-type=FAIL
#SBATCH --output arlb-job_%A_%a.out
#SBATCH --error arlb-job_%A_%a.err

#SBATCH --array=0-9

module purge
module load system singularity

singularity exec --nv singularity_container.sif bash -c "./run_in_container.sh $SLURM_ARRAY_TASK_ID"
