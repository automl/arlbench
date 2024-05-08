#!/bin/bash

directory="procgen_baseline"

mkdir -p "$directory/log"

echo "#!/bin/bash

#SBATCH --ntasks=1
#SBATCH --cpus-per-task=28
#SBATCH --gres=gpu:1
#SBATCH --mem-per-cpu=500M
#SBATCH --job-name procgen_baseline_${1}_${2}
#SBATCH --time 24:00:00
#SBATCH --mail-type fail
#SBATCH --mail-user dierkes@aim.rwth-aachen.de
#SBATCH --partition KathleenG
#SBATCH --output $directory/log/procgen_baseline_${1}_${2}_%A_%a.out
#SBATCH --error $directory/log/procgen_baseline_${1}_${2}_%A_%a.err
#SBATCH --array=0-9

cd ..
source .venv/bin/activate
python runscripts/run_arlbench.py -m --config-name=base "autorl.n_eval_steps=100" "autorl.n_eval_episodes=128"  "autorl.seed=\$SLURM_ARRAY_TASK_ID" "algorithm=$1" "environment=$2" "+sb_zoo=$2_$1"
" > $directory/${1}_${2}.sh
echo "Submitting $directory for $1 on $2"
chmod +x $directory/${1}_${2}.sh
sbatch $directory/${1}_${2}.sh
