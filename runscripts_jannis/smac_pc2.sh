#!/bin/bash

# USAGE run_rs.sh EXPERIMENT      CLUSTER 
# USAGE run_rs.sh cc_cartpole_dqn local  

directory="smac"

mkdir -p "$directory/log"

echo "#!/bin/bash


#SBATCH --cpus-per-task=8
#SBATCH --mem=16GB
#SBATCH -J smac_${1}
#SBATCH -A hpc-prf-intexml                              # TODO check for your project
#SBATCH -t 7-00:00:00                                   # TODO check for your clusters time limit
#SBATCH --mail-type fail
#SBATCH --mail-user becktepe@stud.uni-hannover.de       # TODO enter your mail and hope slurm isn't reaching out to you :D
#SBATCH -p normal                                       # TODO check for your clusters partition
#SBATCH --output $directory/log/smac_${1}_%A.out
#SBATCH --error $directory/log/smac_${1}_%A.err

cd ..
python runscripts/run_arlbench.py -m --config-name=tune_smac "experiments=$1" "cluster=$2" 
" > $directory/${1}.sh
echo "Submitting $directory for $1"
chmod +x $directory/${1}.sh
sbatch $directory/${1}.sh
