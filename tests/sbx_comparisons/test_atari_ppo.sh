!/bin/bash
#SLURM <slurm options>

conda activate arlb

python arlb_ppo.py --n-eval-steps=1250 --training-steps=1250000 --env-framework=envpool --env=Pong-v5 --seed=0 --n-envs-steps=128
