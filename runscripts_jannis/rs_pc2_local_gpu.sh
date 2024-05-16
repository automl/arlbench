#!/bin/bash

# USAGE run_rs.sh EXPERIMENT      
# USAGE run_rs.sh cc_cartpole_dqn  

directory="sobol"

mkdir -p "$directory/log"

echo "#!/bin/bash


#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=16GB
#SBATCH -J arlb_rs_${1}
#SBATCH -A hpc-prf-intexml                              # TODO check for your project
#SBATCH -t 2-00:00:00                                   # TODO check for your clusters time limit
#SBATCH --mail-type fail
#SBATCH --mail-user becktepe@stud.uni-hannover.de       # TODO enter your mail and hope slurm isn't reaching out to you :D
#SBATCH -p gpu                                       # TODO check for your clusters partition
#SBATCH --output $directory/log/arlb_rs_${1}_%A_%a.out
#SBATCH --error $directory/log/arlb_rs_${1}_%A_%a.err
#SBATCH --array=0-9


cd ..
python runscripts/run_arlbench.py -m --config-name=random_runs "autorl.seed=\$SLURM_ARRAY_TASK_ID" "experiments=$1" "cluster=local" 
" > $directory/${1}.sh
echo "Submitting $directory for $1"
chmod +x $directory/${1}.sh
sbatch $directory/${1}.sh
