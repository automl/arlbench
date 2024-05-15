from __future__ import annotations

import argparse
import logging
import os
import sys
import time

import jax
import pandas as pd
from arlbench.core.algorithms import DQN
from arlbench.core.environments import make_env


def dqn(
    dir_name, log, framework, env_name, config, training_kw_args, seed, cnn_policy
):
    env = make_env(
        framework, 
        env_name, 
        n_envs=config["n_envs"], 
        seed=seed, 
        cnn_policy=cnn_policy,
        env_kwargs={
            #"episodic_life": True,
            #"reward_clip": True,
        }
    )
    eval_env = make_env(
        framework, 
        env_name, 
        n_envs=training_kw_args["n_eval_episodes"], 
        seed=seed, 
        cnn_policy=cnn_policy,
        env_kwargs={
            #"episodic_life": False,
            #"reward_clip": False,
        }
    )
    rng = jax.random.PRNGKey(seed)

    log.info(f"JAX devices: {jax.devices()[0].platform.lower()}")
    log.info(f"JAX default backend: {jax.default_backend()}")

    hpo_config = DQN.get_default_hpo_config()
    hpo_config["learning_starts"] = 1
    hpo_config["tau"] = 1.0
    hpo_config["learning_rate"] = 5e-4
    hpo_config["buffer_batch_size"] = 64
    hpo_config["buffer_prio_sampling"] = True
    hpo_config["buffer_alpha"] = 0.6
    hpo_config["buffer_beta"] = 0.6
    hpo_config["buffer_size"] = 1000000
    hpo_config["train_freq"] = 1
    hpo_config["gradient_steps"] = 2
    hpo_config["target_update_interval"] = 250
    nas_config = DQN.get_default_nas_config()
    nas_config["activation"] = "tanh"
    nas_config["hidden_size"] = 256

    agent = DQN(hpo_config, env, eval_env=eval_env, nas_config=nas_config, cnn_policy=cnn_policy)
    algorithm_state = agent.init(rng)

    start = time.time()
    log.info("training started")
    (algorithm_state, results) = agent.train(*algorithm_state, **training_kw_args)
    log.info("training finished")
    training_time = time.time() - start

    mean_return = results.eval_rewards.mean(axis=1)
    std_return = results.eval_rewards.std(axis=1)
    str_results = [
        f"{mean:.2f}+-{std:.2f}"
        for mean, std in zip(mean_return, std_return, strict=False)
    ]
    log.info(f"{training_time}, {str_results}")
    print(f"{training_time}, {str_results}")

    train_info_df = pd.DataFrame()
    for i in range(len(mean_return)):
        train_info_df[f"return_{i}"] = results.eval_rewards[i]

    os.makedirs(
        os.path.join("dqn_results", f"{framework}_{env_name}", dir_name), exist_ok=True
    )
    train_info_df.to_csv(
        os.path.join(
            "dqn_results", f"{framework}_{env_name}", dir_name, f"{seed}_results.csv"
        )
    )
    with open(
        os.path.join(
            "dqn_results", f"{framework}_{env_name}", dir_name, f"{seed}_info"
        ),
        "w",
    ) as f:
        f.write(f"sac_config: {config}\n")
        f.write(f"hpo_config: {hpo_config}\n")
        f.write(f"nas_config: {nas_config}\n")
        f.write(f"time: {training_time}\n")
        f.write(f"returns: {str_results}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir-name", type=str, default="arlb_gpu_2")
    parser.add_argument("--training-steps", type=int, default=100000)
    parser.add_argument("--n-eval-steps", type=int, default=10)
    parser.add_argument("--n-eval-episodes", type=int, default=128)
    parser.add_argument("--n-envs", type=int, default=4)
    parser.add_argument("--seed", type=int, default=9)
    parser.add_argument("--env-framework", type=str, default="envpool")
    parser.add_argument("--env", type=str, default="LunarLander-v2")
    parser.add_argument("--cnn-policy", type=bool, default=False)
    args = parser.parse_args()

    logger = logging.getLogger(__name__)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    stdout_handler = logging.StreamHandler(sys.stdout)
    stdout_handler.setFormatter(formatter)
    logger.setLevel(logging.INFO)
    logger.addHandler(stdout_handler)

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
        dqn(
            dir_name=args.dir_name,
            log=logger,
            framework=args.env_framework,
            env_name=args.env,
            config=config,
            training_kw_args=training_kw_args,
            seed=args.seed,
            cnn_policy=args.cnn_policy,
        )
