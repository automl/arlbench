#!/bin/bash
#for configuration options see: https://uni-paderborn.atlassian.net/wiki/spaces/PC2DOK/pages/12944324/Running+Compute+Jobs
#SBATCH -N 1
#SBATCH -n 4                                
#SBATCH --gres=gpu:a100:1
#SBATCH --mem-per-cpu=4GB
#SBATCH --time=00:05:00

#SBATCH --job-name=arlb_atari
#SBATCH -A hpc-prf-intexml                  

#SBATCH --partition=dgx

#SBATCH --mail-user=becktepe@stud.uni-hannover.de
#SBATCH --mail-type=FAIL
#SBATCH --output arlb_atari-job_%j.out
#SBATCH --error arlb_atari-job_%j.err

module reset
module load system singularity
module load system
module load CUDA

singularity exec singularity_container.sif bash -c "./run_in_container.sh"
