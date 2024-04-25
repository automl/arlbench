. /opt/conda/etc/profile.d/conda.sh
conda activate arlb

cd /tmp/scratch/git_repos/arlbench/tests/sbx_comparisons

# code to run your program
python test_jax.py
# python arlb_dqn.py --training-steps=10000 --n-eval-steps=10 --n-eval-episodes=10 --dir-name=arlb_pc2 --n-envs=10 --env-framework=envpool --env=Pong-v5 --seed=0
