#!/bin/bash
#for configuration options see: https://uni-paderborn.atlassian.net/wiki/spaces/PC2DOK/pages/12944324/Running+Compute+Jobs
#SBATCH -N 1
#SBATCH -n 4
#SBATCH --gres=gpu:a100:1
#SBATCH --mem-per-cpu=4GB
#SBATCH --time=00:05:00

#SBATCH --job-name=cuda_test
#SBATCH -A hpc-prf-intexml

#SBATCH --qos=devel
#SBATCH --partition=dgx

#SBATCH --mail-user=becktepe@stud.uni-hannover.de
#SBATCH --mail-type=FAIL
#SBATCH --output cuda_test.out
#SBATCH --error cuda_test.err

module reset
module load system singularity

singularity exec singularity_container.sif bash -c "./run_in_container.sh"
