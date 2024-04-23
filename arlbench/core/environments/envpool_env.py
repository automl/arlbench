from __future__ import annotations

import functools
from typing import Any

import jax
import numpy as np
import jax.numpy as jnp
import numpy as np

from arlbench.utils import gymnasium_space_to_gymnax_space

from .autorl_env import Environment

ATARI_ENVS = [
    "Adventure-v5",
    "AirRaid-v5",
    "Alien-v5",
    "Amidar-v5",
    "Assault-v5",
    "Asterix-v5",
    "Asteroids-v5",
    "Atlantis-v5",
    "Atlantis2-v5",
    "Backgammon-v5",
    "BankHeist-v5",
    "BasicMath-v5",
    "BattleZone-v5",
    "BeamRider-v5",
    "Berzerk-v5",
    "Blackjack-v5",
    "Bowling-v5",
    "Boxing-v5",
    "Breakout-v5",
    "Carnival-v5",
    "Casino-v5",
    "Centipede-v5",
    "ChopperCommand-v5",
    "CrazyClimber-v5",
    "Crossbow-v5",
    "Darkchambers-v5",
    "Defender-v5",
    "DemonAttack-v5",
    "DonkeyKong-v5",
    "DoubleDunk-v5",
    "Earthworld-v5",
    "ElevatorAction-v5",
    "Enduro-v5",
    "Entombed-v5",
    "Et-v5",
    "FishingDerby-v5",
    "FlagCapture-v5",
    "Freeway-v5",
    "Frogger-v5",
    "Frostbite-v5",
    "Galaxian-v5",
    "Gopher-v5",
    "Gravitar-v5",
    "Hangman-v5",
    "HauntedHouse-v5",
    "Hero-v5",
    "HumanCannonball-v5",
    "IceHockey-v5",
    "Jamesbond-v5",
    "JourneyEscape-v5",
    "Kaboom-v5",
    "Kangaroo-v5",
    "KeystoneKapers-v5",
    "KingKong-v5",
    "Klax-v5",
    "Koolaid-v5",
    "Krull-v5",
    "KungFuMaster-v5",
    "LaserGates-v5",
    "LostLuggage-v5",
    "MarioBros-v5",
    "MiniatureGolf-v5",
    "MontezumaRevenge-v5",
    "MrDo-v5",
    "MsPacman-v5",
    "NameThisGame-v5",
    "Othello-v5",
    "Pacman-v5",
    "Phoenix-v5",
    "Pitfall-v5",
    "Pitfall2-v5",
    "Pong-v5",
    "Pooyan-v5",
    "PrivateEye-v5",
    "Qbert-v5",
    "Riverraid-v5",
    "RoadRunner-v5",
    "Robotank-v5",
    "Seaquest-v5",
    "SirLancelot-v5",
    "Skiing-v5",
    "Solaris-v5",
    "SpaceInvaders-v5",
    "SpaceWar-v5",
    "StarGunner-v5",
    "Superman-v5",
    "Surround-v5",
    "Tennis-v5",
    "Tetris-v5",
    "TicTacToe3d-v5",
    "TimePilot-v5",
    "Trondead-v5",
    "Turmoil-v5",
    "Tutankham-v5",
    "UpNDown-v5",
    "Venture-v5",
    "VideoCheckers-v5",
    "VideoChess-v5",
    "VideoCube-v5",
    "VideoPinball-v5",
    "WizardOfWor-v5",
    "WordZapper-v5",
    "YarsRevenge-v5",
    "Zaxxon-v5",
]


class EnvpoolEnv(Environment):
    def __init__(self, env_name: str, n_envs: int, seed: int):
        import envpool
        env = envpool.make(env_name, env_type="gymnasium", num_envs=n_envs, seed=seed)

        super().__init__(env_name, env, n_envs, seed)

        self.reset_shape = self._env.reset()

        rng = jax.random.key(42)
        _rngs = jax.random.split(rng, self._n_envs)
        self.dummy_action = np.array([
                self.action_space.sample(_rngs[i])
                for i in range(self._n_envs)
        ])
        self.step_shape = self._env.step(self.dummy_action)
        self.single_step_shape = jax.tree_map(lambda x: x[:1], self.step_shape)
        
        self._handle0, _, _, self._xla_step = self._env.xla()

    @functools.partial(jax.jit, static_argnums=0)
    def reset(self, _):
        obs, _ = jax.experimental.io_callback(self._env.reset, self.reset_shape)
        return self._handle0, obs

    @functools.partial(jax.jit, static_argnums=0)
    def step(self, env_state: Any, action: Any, _):
        # todo: update info correctly as well
        self._handle0, (obs, reward, term, trunc, info) = self._xla_step(env_state, action)
        done = jnp.logical_or(term, trunc)

        def reset(obs):
            def reset_idx(i, obs):
                new_obs, _, _, _, _ = jax.experimental.io_callback(self._env.step, self.single_step_shape, action=self.dummy_action[:1], env_id=np.array([i]))
                obs = obs.at[i].set(new_obs[0])
                return obs

            for i in range(self._n_envs):
                obs = jax.lax.cond(
                    done[i],
                    lambda obs: reset_idx(i, obs),
                    lambda obs: obs,
                    obs,
                )
            return obs

        obs = jax.lax.cond(
            jnp.any(done),
            reset,
            lambda obs: obs,
            obs
        )

        return self._handle0, (obs, reward, done, info)

    @property
    def action_space(self):
        return gymnasium_space_to_gymnax_space(self._env.action_space)

    @property
    def observation_space(self):
        return gymnasium_space_to_gymnax_space(self._env.observation_space)
