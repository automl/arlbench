. /opt/conda/etc/profile.d/conda.sh
conda activate arlb

# code to run your program
python arlb_dqn.py --training-steps=125000 --n-eval-steps=125 --n-eval-episodes=10 --dir-name=arlb --n-envs=10 --env-framework=envpool --env=Pong-v5 --seed=0