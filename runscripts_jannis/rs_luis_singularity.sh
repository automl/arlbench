#!/bin/bash

# USAGE run_rs.sh EXPERIMENT      CLUSTER 
# USAGE run_rs.sh cc_cartpole_dqn local  

directory="sobol"

mkdir -p "$directory/log"

# run in container
echo "
. /opt/conda/etc/profile.d/conda.sh
conda activate environment

cd /tmp/bigwork/git_projects/arlbench

python runscripts/run_arlbench.py -m --config-name=random_runs "autorl.seed=\$SLURM_ARRAY_TASK_ID" "+experiments=$1" "cluster=$2" 
" > $directory/${1}_in_container.sh

# start script
echo "#!/bin/bash


#SBATCH --cpus-per-task=8
#SBATCH --mem=16GB
#SBATCH -J arlb_rs_${1}
#SBATCH -t 4-00:00:00                                   # TODO check for your clusters time limit
#SBATCH --mail-type fail
#SBATCH --mail-user becktepe@stud.uni-hannover.de       # TODO enter your mail and hope slurm isn't reaching out to you :D
#SBATCH -p ai                                           # TODO check for your clusters partition
#SBATCH --output $directory/log/arlb_rs_${1}_%A_%a.out
#SBATCH --error $directory/log/arlb_rs_${1}_%A_%a.err
#SBATCH --array=0-0

module purge
module load system singularity

cd ..
singularity exec --bind $BIGWORK:/tmp/bigwork --nv singularity_container.sif bash -c "$directory/${1}_in_container.sh $SLURM_ARRAY_TASK_ID"

/bigwork/nhwpbecj/nhwpbecj/.conda/envs/arlb/bin/
" > $directory/${1}.sh
echo "Submitting $directory for $1"
chmod +x $directory/${1}.sh
sbatch $directory/${1}.sh
