"""Console script for runtime experiments."""
from __future__ import annotations

import sys

import hydra
from codecarbon import track_emissions
import jax
from arlbench.core.environments import make_env
from arlbench.core.algorithms import DQN as ARLBDQN
from arlbench.core.algorithms import PPO as ARLBPPO
from arlbench.core.algorithms import SAC as ARLBSAC
import time
import pandas as pd
import functools
import logging
import gymnax
import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd
from brax import envs as brax_envs
from brax.envs.wrappers.gym import GymWrapper
from arlbench.core.environments import Environment
from gymnax.wrappers.gym import GymnaxToGymWrapper
from datetime import timedelta
from omegaconf import DictConfig, OmegaConf
from envpool.python.protocol import EnvPool
from gymnax.wrappers.purerl import FlattenObservationWrapper, LogWrapper


import jax.numpy as jnp
import os 


def format_time(seconds: float) -> str:
    td = timedelta(seconds=seconds)
    h, m, s = str(td).split(':')
    return f"{int(h):d}:{m:2s}:{s:2s}"


ARLBENCH_ALGORITHMS = {
    "dqn": ARLBDQN,
    "ppo": ARLBPPO,
    "sac": ARLBSAC,
}


def train_arlbench(cfg: DictConfig, logger: logging.Logger):
    env = make_env(
        cfg.environment.framework,
        cfg.environment.name,
        n_envs=cfg.environment.n_envs,
        seed=cfg.seed,
        env_kwargs=cfg.environment.kwargs,
        cnn_policy=cfg.environment.cnn_policy
    )
    eval_env = make_env(
        cfg.environment.framework,
        cfg.environment.name,
        n_envs=cfg.n_eval_episodes,
        seed=cfg.seed,
        env_kwargs=cfg.environment.kwargs,
        cnn_policy=cfg.environment.cnn_policy
    )
    rng = jax.random.PRNGKey(cfg.seed)

    algorithm_cls = ARLBENCH_ALGORITHMS[cfg.algorithm]

    # override NAS config
    nas_config = algorithm_cls.get_default_nas_config()
    if "nas_config" in cfg:
        for k, v in cfg.nas_config:
            nas_config[k] = v

    agent = algorithm_cls(
        cfg.hp_config,
        env,
        nas_config=nas_config,
        eval_env=eval_env,
        cnn_policy=cfg.environment.cnn_policy
    )
    algorithm_state = agent.init(rng)

    logger.info("Training started.")

    start = time.time()
    (algorithm_state, results) = agent.train(
        *algorithm_state,
        n_eval_steps=cfg.n_eval_steps,
        n_eval_episodes=cfg.n_eval_episodes,
        n_total_timesteps=cfg.n_total_timesteps
    )
    training_time = time.time() - start

    logger.info(f"Finished in {format_time(training_time)}s.")

    steps = np.arange(1, cfg.n_eval_steps + 1) * cfg.n_total_timesteps // cfg.n_eval_steps
    returns = results.eval_rewards.mean(axis=1)

    train_info_df = pd.DataFrame({'timesteps': steps, 'returns': returns})

    return train_info_df, training_time


def train_purejaxrl(cfg: DictConfig, logger: logging.Logger):
    if cfg.algorithm == "dqn":
        return train_purejaxrl_dqn(cfg, logger)
    elif cfg.algorithm == "ppo":
        return train_purejaxrl_ppo(cfg, logger)
    else:
        raise ValueError(f"Invalid algorithm for purejaxrl: '{cfg.algorithm}'")


def train_purejaxrl_ppo(
        cfg: DictConfig,
        logger: logging.Logger
    ):
    if cfg.environment.framework == "gymnax":
        env, env_params = gymnax.make(cfg.environment.name)

        if isinstance(env.action_space(env_params), gymnax.spaces.Discrete):
            from purejaxrl.purejaxrl.ppo import make_train
        elif isinstance(env.action_space(env_params), gymnax.spaces.Box):
            from purejaxrl.purejaxrl.ppo_continuous_action import make_train
        else:
            raise ValueError(f"Invalid action space type: {type(env.action_space)}.")
    else:
        # brax only has continuous action spaces
        from purejaxrl.purejaxrl.ppo_continuous_action import make_train
    
    # TODO
    start = time.time()
    # train_func = jax.jit()
    # result = train_func()
    training_time = time.time() - start

    return pd.DataFrame([]), training_time


def train_purejaxrl_dqn(cfg: DictConfig, logger: logging.Logger):
    # this is a bit hacky but it allows us to access the PureJaxRL submodule 
    root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../purejaxrl'))
    sys.path.append(root_dir)
    from purejaxrl.dqn import make_train

    config = {
        "NUM_ENVS": cfg.environment.n_envs,
        "BUFFER_SIZE": cfg.hp_config.buffer_size,
        "BUFFER_BATCH_SIZE": cfg.hp_config.buffer_batch_size,
        "TOTAL_TIMESTEPS": cfg.n_total_timesteps,
        "EPSILON_START": cfg.hp_config.epsilon,
        "EPSILON_FINISH": cfg.hp_config.epsilon,
        "EPSILON_ANNEAL_TIME": 1,
        "TARGET_UPDATE_INTERVAL": cfg.hp_config.target_network_update_freq,
        "LR": cfg.hp_config.lr,
        "LEARNING_STARTS": cfg.hp_config.learning_starts,
        "TRAINING_INTERVAL": cfg.hp_config.train_frequency,
        "LR_LINEAR_DECAY": False,
        "GAMMA": cfg.hp_config.gamma,
        "TAU": cfg.hp_config.tau,
        "ENV_NAME": cfg.environment.name,
        "SEED": cfg.seed,
        "NUM_SEEDS": 1,
        "WANDB_MODE": "disabled",  # set to online to activate wandb
        "ENTITY": "",
        "PROJECT": "",
    }

    rng = jax.random.PRNGKey(cfg.seed)
    rngs = jax.random.split(rng, 1)

    logger.info("Training started.")

    start = time.time()
    train_vjit = jax.jit(jax.vmap(make_train(config)))
    outs = jax.block_until_ready(train_vjit(rngs))
    training_time = time.time() - start

    logging.info(f"Finished in {format_time(training_time)}s.")
    print(outs.keys())

    train_info_df = pd.DataFrame({'timesteps': outs["metrics"]["timesteps"][0], 'returns': outs["metrics"]["returns"][0]})

    return train_info_df, training_time


def train_sbx(cfg: DictConfig, logger: logging.Logger):
    from stable_baselines3.common.callbacks import BaseCallback
    from stable_baselines3.common.evaluation import evaluate_policy
    from stable_baselines3.common.vec_env import VecEnvWrapper, VecMonitor
    from stable_baselines3.common.vec_env.base_vec_env import VecEnvObs, VecEnvStepReturn
    from sbx import DQN, PPO, SAC
    from jax import nn

    class VecAdapter(VecEnvWrapper):
        """Convert EnvPool object to a Stable-Baselines3 (SB3) VecEnv.

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

        def seed(self, seed: int | None = None) -> None:
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
                        for key in info_dict
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
            eval_env,
            eval_freq,
            n_eval_episodes,
            seed,
        ):
            super().__init__(verbose=0)

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

            initial_state = (env_state, jnp.full((1), 0.0), jnp.full((1), False))

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
            # n_evals = int(np.ceil(num_eval_episodes / self.eval_env.n_envs))
            (self.rng, _), returns = jax.lax.scan(
                self._env_episode,
                (self.rng, self.model.policy.actor_state.params),
                None,
                num_eval_episodes,
            )
            return jnp.concat(returns)[:num_eval_episodes]

        def _on_step(self) -> bool:
            if self.eval_freq > 0 and self.n_calls % self.eval_freq == 0:
                # returns = self.eval(self.n_eval_episodes)
                returns, _ = evaluate_policy(
                    self.model,
                    self.eval_env,
                    n_eval_episodes=self.n_eval_episodes,
                    return_episode_rewards=True,
                )
                self.return_list.append(returns)
                jax.debug.print("{returns}", returns=np.array(returns).mean())
            return True

        def _on_training_end(self) -> None:
            pass
            # returns, _ = evaluate_policy(
            #    self.model, self.eval_env, n_eval_episodes=self.n_eval_episodes, return_episode_rewards=True
            # )
            # self.return_list.append(returns)

        def get_returns(self):
            return self.return_list

    import envpool

    env = VecMonitor(
        VecAdapter(
            envpool.make(
                cfg.environment.name,
                env_type="gymnasium",
                num_envs=cfg.environment.n_envs,
                seed=cfg.seed,
                **cfg.environment.kwargs
            )
        )
    )
    eval_env = VecMonitor(
        VecAdapter(
            envpool.make(
                cfg.environment.name,
                env_type="gymnasium",
                num_envs=cfg.n_eval_episodes,
                seed=cfg.seed)
        )
    )

    eval_callback = EvalTrainingMetricsCallback(
        eval_env=eval_env,
        eval_freq=cfg.n_total_timesteps // cfg.n_eval_steps // cfg.environment.n_envs,
        n_eval_episodes=cfg.n_eval_episodes,
        seed=cfg.seed,
    )

    if cfg.algorithm == "dqn":
        nas_config = {"net_arch": [350, 350], "activation_fn": nn.relu}
        model = DQN(
            "CnnPolicy" if cfg.environment.cnn_policy else "MlpPolicy",
            env,
            policy_kwargs=nas_config,
            verbose=4,
            seed=cfg.seed,
            learning_starts=cfg.hp_config.learning_starts,
            target_update_interval=cfg.hp_config.target_network_update_freq,
            exploration_final_eps=cfg.hp_config.epsilon,
            exploration_initial_eps=cfg.hp_config.epsilon,
            gradient_steps=cfg.hp_config.gradient_steps,
            buffer_size=cfg.hp_config.buffer_size,
            learning_rate=cfg.hp_config.lr,
            batch_size=cfg.hp_config.buffer_batch_size,
            train_freq=cfg.hp_config.train_frequency,
            tau=cfg.hp_config.tau,
            gamma=cfg.hp_config.gamma
        )
    elif cfg.algorithm == "ppo":
        nas_config = {"net_arch": [350, 350], "activation_fn": nn.relu}
        model = PPO(
            "CnnPolicy" if cfg.environment.cnn_policy else "MlpPolicy",
            env,
            policy_kwargs=nas_config,
            verbose=4,
            seed=cfg.seed,
            clip_range=cfg.hp_config.clip_eps,
            ent_coef=cfg.hp_config.ent_coef,
            gae_lambda=cfg.hp_config.gae_lambda,
            gamma=cfg.hp_config.gamma,
            learning_rate=cfg.hp_config.lr,
            max_grad_norm=cfg.hp_config.max_grad_norm,
            batch_size=cfg.hp_config.minibatch_size,
            n_steps=cfg.hp_config.n_steps,
            n_epochs=cfg.hp_config.update_epochs,
            vf_coef=cfg.hp_config.vf_coef
        )
    elif cfg.algorithm == "sac":
        nas_config = {"net_arch": [256, 256]}
        model = SAC(
            "CnnPolicy" if cfg.environment.cnn_policy else "MlpPolicy",
            env,
            policy_kwargs=nas_config,
            verbose=4,
            seed=cfg.seed,
            batch_size=cfg.hp_config.buffer_batch_size,
            buffer_size=cfg.hp_config.buffer_size,
            gamma=cfg.hp_config.gamma,
            gradient_steps=cfg.hp_config.gradient_steps,
            learning_starts=cfg.hp_config.learning_starts,
            learning_rate=cfg.hp_config.lr,
            train_freq=cfg.hp_config.train_frequency
        )
    else: 
        raise ValueError(f"Invalid algorithm: {cfg.algorithm}.")

    start = time.time()
    model.learn(
        total_timesteps=int(cfg.n_total_timesteps), callback=eval_callback
    )
    training_time = time.time() - start

    logging.info(f"Finished in {format_time(training_time)}s.")

    eval_returns = np.array(eval_callback.get_returns())

    steps = np.arange(1, cfg.n_eval_steps + 1) * cfg.n_total_timesteps // cfg.n_eval_steps
    returns = eval_returns.mean(axis=1)

    train_info_df = pd.DataFrame({'timesteps': steps, 'returns': returns})

    return train_info_df, training_time


@hydra.main(version_base=None, config_path="configs", config_name="runtime_experiments")
# @track_emissions(offline=True, country_iso_code="DEU")
def main(cfg: DictConfig):
    print(OmegaConf.to_yaml(cfg))

    dir_name= "./"      # TODO add to config
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    if "algorithm_framework" not in cfg:
        raise ValueError("Key 'algorithm_framework' not in config.")

    if cfg.algorithm_framework == "arlbench":
        train_info_df, training_time = train_arlbench(cfg, logger)
    elif cfg.algorithm_framework == "purejaxrl":
        if cfg.environment.framework != "gymnax":
            raise ValueError("Only gymnax is supported as environment framework for purejaxrl.")
        
        train_info_df, training_time = train_purejaxrl(cfg, logger)
    elif cfg.algorithm_framework == "sbx":
        if cfg.environment.framework != "envpool":
            raise ValueError("Only envpool is supported as environment framework for SBX.")
        
        train_info_df, training_time = train_sbx(cfg, logger)
    else:
        raise ValueError(f"Invalid value for 'algorithm_framework': '{cfg.algorithm_framework}'.")
    
    train_info_df.to_csv("evaluation.csv", index=False)

    with open("info", "w",) as f:
        f.write(f"config: {str(cfg)}\n")
        f.write(f"time: {training_time}\n")


if __name__ == "__main__":
    sys.exit(main())  # pragma: no cover