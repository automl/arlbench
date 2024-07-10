#!/bin/bash

# USAGE run_rs.sh ALGORITHM ENVIRONMENT CLUSTER SEARCH_SPACE

directory="runtime"

mkdir -p "$directory/log"

echo "#!/bin/bash

#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=2000M                                      
#SBATCH --job-name=rt_$3
#SBATCH --account=p0021208
#SBATCH --gres=gpu:1
#SBATCH --partition=c23g
#SBATCH -t 24:00:00            
#SBATCH --mail-type fail                                
#SBATCH --mail-user dierkes@aim.rwth-aachen.de
#SBATCH --output $directory/log/arlb_rs_${1}_${2}_%A_%a.out
#SBATCH --error $directory/log/arlb_rs_${1}_${2}_%A_%a.err
#SBATCH --array 0-9

cd ..
source /rwthfs/rz/cluster/home/oh751555/i14/arlbench/.venv/bin/activate 
python runscripts/run_runtime_experiments.py "algorithm_framework=$4" "seed=\$SLURM_ARRAY_TASK_ID" "+sb_zoo=$2_$3_$1" "environment.framework=envpool"
" > $directory/${1}_${2}.sh
echo "Submitting $directory for $1 on $2"
chmod +x $directory/${1}_${2}.sh
sbatch $directory/${1}_${2}.sh
