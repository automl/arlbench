#!/bin/bash

# USAGE run_rs.sh EXPERIMENT      CLUSTER 
# USAGE run_rs.sh cc_cartpole_dqn local  

directory="rs"

mkdir -p "$directory/log"

echo "#!/bin/bash


#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=2000M
#SBATCH --job-name=rs_${1}
#SBATCH --partition=c23mm
#SBATCH --account=p0021208
#SBATCH -t 12:00:00                                 
#SBATCH --mail-type fail
#SBATCH --mail-user dierkes.julian@rwth-aachen.de
#SBATCH --output $directory/log/rs_${1}_%A.out
#SBATCH --error $directory/log/rs_${1}_%A.err
#SBATCH --array=42-46

cd ..
source /rwthfs/rz/cluster/home/oh751555/i14/arlbench/.venv/bin/activate
python runscripts/run_arlbench.py -m --config-name=tune_rs "experiments=$1" "cluster=$2" "search_space.seed=\$SLURM_ARRAY_TASK_ID"
" > $directory/${1}.sh
echo "Submitting $directory for $1"
chmod +x $directory/${1}.sh
sbatch --begin=now $directory/${1}.sh
