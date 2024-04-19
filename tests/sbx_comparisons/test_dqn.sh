for i in {1..10}; do
    python arlb_dqn.py --training-steps=1000000 --n-eval-steps=100 --n-eval-episodes=10 --dir-name=arlb_test --n-envs=10 --env-framework=gymnax --env=CartPole-v1 --seed=$i
done