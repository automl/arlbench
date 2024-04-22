import argparse
import logging
import os
import time

import jax
import pandas as pd

from arlbench.core.algorithms import PPO
from arlbench.core.environments import make_env


def ppo_runner(dir_name, log, framework, env_name, config, training_kw_args, seed, cnn_policy):
    env = make_env(framework, env_name, n_envs=config["n_envs"], seed=seed)
    rng = jax.random.PRNGKey(seed)

    # OrderedDict([('batch_size', 256),
    #          ('clip_range', 'lin_0.1'),
    #          ('ent_coef', 0.01),
    #          ('frame_stack', 4),
    #          ('n_timesteps', 10000000.0),

    #    "minibatch_size": Integer("minibatch_size", (4, 1024), default=64),
    #     "lr": Float("lr", (1e-5, 0.1), default=2.5e-4),
    #     "n_steps": Integer("n_steps", (1, 10000), default=1024),
    #     "update_epochs": Integer("update_epochs", (1, int(1e5)), default=10),
    #     "activation": Categorical("activation", ["tanh", "relu"], default="tanh"),
    #     "gamma": Float("gamma", (0., 1.), default=0.99),
    #     "gae_lambda": Float("gae_lambda", (0., 1.), default=0.95),
    #     "clip_eps": Float("clip_eps", (0., 1.), default=0.2),
    #     "ent_coef": Float("ent_coef", (0., 1.), default=0.01),
    #     "vf_coef": Float("vf_coef", (0., 1.), default=0.5),
    #     "max_grad_norm": Float("max_grad_norm", (0., 10.), default=0.5)

    hpo_config = PPO.get_default_hpo_config()
    hpo_config["minibatch_size"] = 256
    hpo_config["update_epochs"] = 4
    hpo_config["gamma"] = 0.99
    hpo_config["gae_lambda"] = 0.95
    hpo_config["ent_coef"] = 0.01
    hpo_config["max_grad_norm"] = 0.5

    nas_config = PPO.get_default_nas_config()
    nas_config["activation"] = "relu"
    nas_config["hidden_size"] = 256

    agent = PPO(hpo_config, env, nas_config=nas_config, cnn_policy=cnn_policy)
    algorithm_state = agent.init(rng)

    start = time.time()
    log.info(f"training started")
    algorithm_state, result = agent.train(*algorithm_state, **training_kw_args)
    log.info(f"training finished")
    training_time = time.time() - start

    mean_return = result.eval_rewards.mean(axis=1)
    std_return = result.eval_rewards.std(axis=1)
    str_results = [f"{mean:.2f}+-{std:.2f}" for mean, std in zip(mean_return, std_return)]
    log.info(f"{training_time}, {str_results}")

    train_info_df = pd.DataFrame()
    for i in range(len(mean_return)):
        train_info_df[f"return_{i}"] = result.eval_rewards[i]

    os.makedirs(os.path.join("ppo_results", f"{framework}_{env_name}", dir_name), exist_ok=True)
    train_info_df.to_csv(os.path.join("ppo_results", f"{framework}_{env_name}", dir_name, f"{seed}_results.csv"))
    with open(os.path.join("ppo_results", f"{framework}_{env_name}", dir_name, f"{seed}_info"), "w") as f:
        f.write(f"ppo_config: {config}\n")
        f.write(f"hpo_config: {hpo_config}\n")
        f.write(f"nas_config: {nas_config}\n")
        f.write(f"time: {training_time}\n")
        f.write(f"returns: {str_results}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir-name", type=str, default="test")
    parser.add_argument("--training-steps", type=int, default=100000)
    parser.add_argument("--n-eval-steps", type=int, default=100)
    parser.add_argument("--n-eval-episodes", type=int, default=10)
    parser.add_argument("--n-envs", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--env-framework", type=str, default="gymnax")
    parser.add_argument("--env", type=str, default="CartPole-v1")
    parser.add_argument("--n-env-steps", type=int, default=500)
    parser.add_argument("--cnn-policy", type=bool, default=False)
    args = parser.parse_args()

    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)

    training_kw_args = {
        "n_total_timesteps": args.training_steps,
        "n_eval_steps": args.n_eval_steps,
        "n_eval_episodes": args.n_eval_episodes,
    }

    config = {
        "n_envs": args.n_envs,
        "track_metrics": False,
        "track_traj": False,
    }

    with jax.disable_jit(disable=False):
        ppo_runner(
            dir_name=args.dir_name,
            log=logger,
            framework=args.env_framework,
            env_name=args.env,
            config=config,
            training_kw_args=training_kw_args,
            seed=args.seed,
            cnn_policy=args.cnn_policy
        )