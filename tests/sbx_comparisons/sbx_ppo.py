import argparse
import functools
import logging
import os
import time
from typing import Optional

import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd
from brax import envs
from brax.envs.wrappers.gym import GymWrapper
from sbx import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.vec_env import VecEnvWrapper, VecMonitor
from stable_baselines3.common.vec_env.base_vec_env import (
  VecEnvObs,
  VecEnvStepReturn,
)
import envpool
from envpool.python.protocol import EnvPool


class VecAdapter(VecEnvWrapper):
  """
  Convert EnvPool object to a Stable-Baselines3 (SB3) VecEnv.

  :param venv: The envpool object.
  """

  def __init__(self, venv: EnvPool):
    # Retrieve the number of environments from the config
    venv.num_envs = venv.spec.config.num_envs
    super().__init__(venv=venv)

  def step_async(self, actions: np.ndarray) -> None:
    self.actions = actions

  def reset(self) -> VecEnvObs:
    return self.venv.reset()[0]

  def seed(self, seed: Optional[int] = None) -> None:
    # You can only seed EnvPool env by calling envpool.make()
    pass

  def step_wait(self) -> VecEnvStepReturn:
    obs, rewards, terms, truncs, info_dict = self.venv.step(self.actions)
    dones = terms + truncs
    infos = []
    # Convert dict to list of dict
    # and add terminal observation
    for i in range(self.num_envs):
      infos.append(
        {
          key: info_dict[key][i]
          for key in info_dict.keys()
          if isinstance(info_dict[key], np.ndarray)
        }
      )
      if dones[i]:
        infos[i]["terminal_observation"] = obs[i]
        obs[i] = self.venv.reset(np.array([i]))[0]
    return obs, rewards, dones, infos


class EvalTrainingMetricsCallback(BaseCallback):
    def __init__(
            self,
            framework,
            eval_env,
            eval_freq,
            n_eval_episodes,
            seed,
    ):
        super().__init__(verbose=0)

        self.framework = framework
        self.eval_env = eval_env
        self.eval_freq = eval_freq
        self.n_eval_episodes = n_eval_episodes
        self.return_list = []
        self.rng = jax.random.PRNGKey(seed)

    @functools.partial(jax.jit, static_argnums=0)
    def _env_episode(self, carry, _):
        rng, actor_params = carry
        rng, reset_rng = jax.random.split(rng)

        env_state = self.eval_env.reset(reset_rng)

        initial_state = (
            env_state,
            jnp.full((1), 0.),
            jnp.full((1), False)
        )

        def cond_fn(carry):
            state, ret, done = carry
            return jnp.logical_not(jnp.all(done))

        def body_fn(carry):
            state, ret, done = carry
            obs = jnp.expand_dims(state.obs, axis=0)
            action = self.model.policy.actor_state.apply_fn(actor_params, obs).mode()
            state = self.eval_env.step(state, action[0])

            # Count rewards only for envs that are not already done
            ret += state.reward * ~done

            done = jnp.logical_or(done, jnp.bool(state.done))

            return (state, ret, done)

        final_state = jax.lax.while_loop(cond_fn, body_fn, initial_state)
        _, returns, _ = final_state

        return (rng, actor_params), returns

    def eval(self, num_eval_episodes):
        # Number of parallel evaluations, each with n_envs environments
        #n_evals = int(np.ceil(num_eval_episodes / self.eval_env.n_envs))
        (self.rng, _), returns = jax.lax.scan(
            self._env_episode, (self.rng, self.model.policy.actor_state.params), None, num_eval_episodes
        )
        return jnp.concat(returns)[:num_eval_episodes]

    def _on_step(self) -> bool:
        if self.eval_freq > 0 and self.num_timesteps % self.eval_freq == 0:
            if self.framework == "brax":
                returns = self.eval(self.n_eval_episodes)
            else:
                returns, lengths = evaluate_policy(
                    self.model, self.eval_env, n_eval_episodes=self.n_eval_episodes, return_episode_rewards=True
                )
                returns = np.array(returns)
            self.return_list.append(returns)
            jax.debug.print("{returns}", returns=returns.mean())
        return True

    def _on_training_end(self) -> None:
        a = 0
        #returns, _ = evaluate_policy(
        #    self.model, self.eval_env, n_eval_episodes=self.n_eval_episodes, return_episode_rewards=True
        #)
        #self.return_list.append(returns)

    def get_returns(self):
        return self.return_list


def ppo_runner(dir_name, log, framework, env_name, ppo_config, seed):

    if framework == "brax":
        env = envs.create(env_name, backend="spring", episode_length=1000)
        env = GymWrapper(env)
        eval_env = envs.create(env_name, backend="spring", episode_length=1000)
    if framework == "envpool":
        import envpool
        env = VecMonitor(VecAdapter(envpool.make(env_name, env_type="gymnasium", num_envs=ppo_config["n_envs"], seed=seed)))
        eval_env = VecMonitor(VecAdapter(envpool.make(env_name, env_type="gymnasium", num_envs=128, seed=seed)))
    if framework == "gymnax":
        import gymnax
        from gymnax.wrappers.gym import GymnaxToGymWrapper
        env, env_params = gymnax.make(env_name)
        env = GymnaxToGymWrapper(env)
        eval_env, eval_env_params = gymnax.make(env_name)
        eval_env = GymnaxToGymWrapper(eval_env)


    eval_callback = EvalTrainingMetricsCallback(
        framework=framework, eval_env=eval_env, eval_freq=ppo_config["eval_freq"], n_eval_episodes=128, seed=seed
    )

    hpo_config = {}
    nas_config = dict(net_arch=[256, 256])
    model = PPO(
        "MlpPolicy", 
        env, 
        policy_kwargs=nas_config, 
        verbose=4, 
        seed=seed,
        batch_size=256,
        n_steps=32,
        gamma=0.98,
        learning_rate=1e-3,
        n_epochs=20,
    )

    start = time.time()
    model.learn(total_timesteps=int(ppo_config["n_total_timesteps"]), callback=eval_callback)
    training_time = time.time() - start

    eval_returns = np.array(eval_callback.get_returns())
    mean_return = eval_returns.mean(axis=1)
    std_return = eval_returns.std(axis=1)
    str_results = [f"{mean:.2f}+-{std:.2f}" for mean, std in zip(mean_return, std_return)]
    log.info(f"{training_time}, {str_results}")

    train_info_df = pd.DataFrame()
    for i in range(len(mean_return)):
        train_info_df[f"return_{i}"] = eval_returns[i]

    os.makedirs(os.path.join("ppo_results", f"{framework}_{env_name}", dir_name), exist_ok=True)
    train_info_df.to_csv(os.path.join("ppo_results", f"{framework}_{env_name}", dir_name, f"{seed}_results.csv"))
    with open(os.path.join("ppo_results", f"{framework}_{env_name}", dir_name, f"{seed}_info"), "w") as f:
        f.write(f"ppo_config: {ppo_config}\n")
        f.write(f"hpo_config: {hpo_config}\n")
        f.write(f"nas_config: {nas_config}\n")
        f.write(f"time: {training_time}\n")
        f.write(f"returns: {str_results}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir-name", type=str, default="test")
    parser.add_argument("--training-steps", type=int, default=1000000)
    parser.add_argument("--eval-freq", type=int, default=1000)
    parser.add_argument("--n-eval-episodes", type=int, default=10)
    parser.add_argument("--n-envs", type=int, default=16)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--env-framework", type=str, default="gymnax")
    parser.add_argument("--env", type=str, default="CartPole-v1")
    parser.add_argument("--n-env-steps", type=int, default=500)
    args = parser.parse_args()

    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)

    ppo_config = {
        "n_total_timesteps": args.training_steps,
        "n_envs": args.n_envs,
        "n_env_steps": args.n_env_steps,
        "eval_freq": args.eval_freq,
        "n_eval_episodes": args.n_eval_episodes,
        "track_metrics": False,
        "track_traj": False,
    }

    with jax.disable_jit(disable=False):
        ppo_runner(
            dir_name=args.dir_name,
            log=logger,
            framework=args.env_framework,
            env_name=args.env,
            ppo_config=ppo_config,
            seed=args.seed,
        )
