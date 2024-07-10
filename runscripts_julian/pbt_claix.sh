#!/bin/bash

# USAGE run_rs.sh EXPERIMENT      CLUSTER 
# USAGE run_rs.sh cc_cartpole_dqn local  

directory="pbt"

mkdir -p "$directory/log"

echo "#!/bin/bash

#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=4000M
#SBATCH --job-name=pbt_${1}
#SBATCH --account=p0021208
#SBATCH --partition=c23ms
#SBATCH --time 96:00:00                                 
#SBATCH --mail-type fail
#SBATCH --mail-user dierkes.julian@rwth-aachen.de
#SBATCH --output $directory/log/pbt_${1}_%A.out
#SBATCH --error $directory/log/pbt_${1}_%A.err
#SBATCH --array 0-4%2

cd ..
source /rwthfs/rz/cluster/home/oh751555/i14/arlbench/.venv/bin/activate
python runscripts/run_arlbench.py -m --config-name=tune_pbt "experiments=$1" "cluster=$2" "pbt_seed=\$SLURM_ARRAY_TASK_ID" "search_space.seed=\$SLURM_ARRAY_TASK_ID"
" > $directory/${1}.sh
echo "Submitting $directory for $1"
chmod +x $directory/${1}.sh
sbatch --begin=now $directory/${1}.sh
