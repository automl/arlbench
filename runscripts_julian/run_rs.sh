#!/bin/bash

# USAGE run_rs.sh ALGORITHM ENVIRONMENT CLUSTER SEARCH_SPACE

directory="sobol"

mkdir -p "$directory/log"

echo "#!/bin/bash

#SBATCH --cpus-per-task=4 
#SBATCH --mem-per-cpu=2000M                                      
#SBATCH --job-name=rs_atari
###SBATCH --account=
#SBATCH -t 128:00:00            
#SBATCH --mail-type fail                                
#SBATCH --mail-user dierkes@aim.rwth-aachen.de
#SBATCH --output $directory/log/arlb_rs_${1}_${2}_%A_%a.out
#SBATCH --error $directory/log/arlb_rs_${1}_${2}_%A_%a.err
#SBATCH --array 0-6%2

cd ..
source /rwthfs/rz/cluster/home/oh751555/i14/arlbench/.venv/bin/activate
python runscripts/run_arlbench.py -m --config-name=random_runs "autorl.seed=\$SLURM_ARRAY_TASK_ID" "algorithm=$1" "search_space=$1_8env" "environment=$2" "cluster=$3" "hydra.sweeper.n_trials=$4" "hydra.sweeper.sweeper_kwargs.max_parallelization=$5" "hydra.sweeper.sweeper_kwargs.job_array_size_limit=256" "hydra.launcher.array_parallelism=$6" "+experiments=$2_$1" 
" > $directory/${1}_${2}.sh
echo "Submitting $directory for $1 on $2"
chmod +x $directory/${1}_${2}.sh
sbatch $directory/${1}_${2}.sh
