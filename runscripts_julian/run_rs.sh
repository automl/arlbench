#!/bin/bash

# USAGE run_rs.sh ALGORITHM ENVIRONMENT CLUSTER
# USAGE run_rs.sh dqn cc_cartpole local

directory="sobol"

mkdir -p "$directory/log"
#!/bin/bash

# USAGE run_rs.sh ALGORITHM ENVIRONMENT CLUSTER SEARCH_SPACE
# USAGE run_rs.sh dqn       cc_cartpole local   dqn_cc

directory="sobol"

mkdir -p "$directory/log"

echo "#!/bin/bash


#SBATCH --cpus-per-task=8
#SBATCH --mem=16GB
#SBATCH -J arlb_rs_${1}_${2}
#SBATCH -A hpc-prf-intexml                              # TODO check for your project
#SBATCH -t 7-00:00:00                                   # TODO check for your clusters time limit
#SBATCH --mail-type fail
#SBATCH --mail-user becktepe@stud.uni-hannover.de       # TODO enter your mail and hope slurm isn't reaching out to you :D
#SBATCH -p normal                                       # TODO check for your clusters partition
#SBATCH --output $directory/log/arlb_rs_${1}_${2}_%A_%a.out
#SBATCH --error $directory/log/arlb_rs_${1}_${2}_%A_%a.err
#SBATCH --array=0-9


cd ..
python runscripts/run_arlbench.py -m --config-name=random_runs "autorl.seed=\$SLURM_ARRAY_TASK_ID" "algorithm=$1" "search_space=$1" "environment=$2" "cluster=$3" "search_space=$4"
" > $directory/${1}_${2}.sh
echo "Submitting $directory for $1 on $2"
chmod +x $directory/${1}_${2}.sh
sbatch $directory/${1}_${2}.sh

echo "#!/bin/bash

#SBATCH --cpus-per-task=4 
#SBATCH --mem-per-cpu=2000M                                      
#SBATCH --job-name=rs_atari
###SBATCH --account=
#SBATCH -t 128:00:00                                   # TODO check for your clusters time limit
#SBATCH --mail-type fail                                
#SBATCH --mail-user becktepe@stud.uni-hannover.de       # TODO enter your mail and hope slurm isn't reaching out to you :D
#SBATCH --output $directory/log/arlb_rs_${1}_${2}_%A_%a.out
#SBATCH --error $directory/log/arlb_rs_${1}_${2}_%A_%a.err
#SBATCH --array=0-9

cd ..
source /rwthfs/rz/cluster/home/oh751555/i14/arlbench/.venv/bin/activate
python runscripts/run_arlbench.py -m --config-name=random_runs "autorl.seed=\$SLURM_ARRAY_TASK_ID" "algorithm=$1" "search_space=$1" "environment=$2" "cluster=$3"
" > $directory/${1}_${2}.sh
echo "Submitting $directory for $1 on $2"
chmod +x $directory/${1}_${2}.sh
sbatch $directory/${1}_${2}.sh
