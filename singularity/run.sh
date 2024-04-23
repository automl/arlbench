#!/bin/bash
#for configuration options see: https://uni-paderborn.atlassian.net/wiki/spaces/PC2DOK/pages/12944324/Running+Compute+Jobs
#SBATCH -N 1
#SBATCH -n 4                                    # Number of CPUs you want on that machine (<=128)
#SBATCH --mem 4GB                               # Amount of RAM you want (<=239GB), e.g. 64GB
#SBATCH -J <JOBNAME>                            # Name of your job - adjust as you like
#SBATCH -A hpc-prf-intexml                      # Project name, do not change
#SBATCH -t 00:05:00                             # Timelimit of the job. See documentation for format.
#SBATCH --mail-type fail                        # Send an email, if the job fails.
#SBATCH --mail-user <MAIL>                      # Where to send the email to
#SBATCH -p normal                               # Normal job, no GPUs

#SBATCH --array=0-1                             # Only use this if you want the cluster to automaticall start multiple jobs, which all run the script below, but with a different argument. See also: https://support.ceci-hpc.be/doc/_contents/QuickStart/SubmittingJobs/SlurmTutorial.html#embarrassingly-parallel-workload-example-job-array

module reset
module load system singularity

singularity exec singularity_container.sif bash -c "./run_in_container.sh $SLURM_ARRAY_TASK_ID"
