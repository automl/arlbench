#!/bin/bash

# USAGE run_rs.sh EXPERIMENT      
# USAGE run_rs.sh cc_cartpole_dqn  

directory="pbt"

mkdir -p "$directory/log"

echo "#!/bin/bash


#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:1
#SBATCH --mem=64GB
#SBATCH -J pbt_${1}
#SBATCH -t 4-00:00:00                                   # TODO check for your clusters time limit
#SBATCH --mail-type fail
#SBATCH --mail-user becktepe@stud.uni-hannover.de       # TODO enter your mail and hope slurm isn't reaching out to you :D
#SBATCH -p ai,tnt                                              # TODO check for your clusters partition
#SBATCH --output $directory/log/pbt_${1}_%A_%a.out
#SBATCH --error $directory/log/pbt_${1}_%A_%a.err
#SBATCH --array=42-42

module load Miniconda3
conda activate /bigwork/nhwpbecj/nhwpbecj/.conda/envs/arlb
module load CUDA
export CUDA_VISIBLE_DEVICES=0


cd ..
python runscripts/run_arlbench.py -m --config-name=tune_pbt "experiments=$1" "cluster=local" "pbt_seed=\$SLURM_ARRAY_TASK_ID" 
" > $directory/${1}.sh
echo "Submitting $directory for $1"
chmod +x $directory/${1}.sh
sbatch $directory/${1}.sh
