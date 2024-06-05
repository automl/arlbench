from __future__ import annotations

import functools
from typing import Any

import jax
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


def numpy_to_jax(x: np.ndarray) -> jnp.ndarray:
    """Converts numpy arrays to jax numpy arrays."""
    if isinstance(x, np.ndarray):
        return jnp.array(x)
    else:
        return x


class EnvpoolEnv(Environment):
    """An envpool-based RL environment."""

    def __init__(
        self,
        env_name: str,
        n_envs: int,
        seed: int,
        env_kwargs: dict[str, Any] | None = None,
    ):
        """Creates an envpool environment for JAX-based RL training.

        Args:
            env_name (str): Name/id of the brax environment.
            n_envs (int): Number of environments.
            seed (int): Random seed.
            env_kwargs (dict[str, Any] | None, optional): Keyword arguments to pass to the brax environment. Defaults to None.
        """
        if env_kwargs is None:
            env_kwargs = {}
        try:
            import envpool
        except ImportError:
            raise ValueError(
                "Failed to import envpool. Please make sure that the package is installed."
            )
        env = envpool.make(
            env_name, env_type="gymnasium", num_envs=n_envs, seed=seed, **env_kwargs
        )

        # We need this since Atari has some special cases to consider
        self.is_atari = env_name in ATARI_ENVS
        if self.is_atari:
            self.episodic_life = env_kwargs.get("episodic_life", False)
            self.use_fire_reset = env_kwargs.get("use_fire_reset", True)

        super().__init__(env_name, env, n_envs, seed)

        # For JAX IO callbacks we need the shape of the reset function
        self.reset_shape = self._reset()

        if self.is_atari:
            self.has_lives = jnp.all(self.reset_shape[1]["lives"] > 0)

        # The dummy actions are used for Atari games to select a random strategy
        # in the beginning
        rng = jax.random.key(42)
        _rngs = jax.random.split(rng, self._n_envs)
        self.dummy_action = np.array(
            [self.action_space.sample(_rngs[i]) for i in range(self._n_envs)]
        )

        if self.is_atari:
            # fire action for reset
            self.dummy_action = np.ones_like(self.dummy_action)

        # These shapes are also required for the IO callbacks
        self.step_shape = self._step(self.dummy_action)
        self.single_step_shape = jax.tree_map(lambda x: x[:1], self.step_shape)

        # This gives us the envpool XLA interface
        # See https://envpool.readthedocs.io/en/latest/content/xla_interface.html
        self._handle0, self.recv, self.send, self._xla_step = self._env.xla()

    def _reset(self) -> tuple[jnp.ndarray, jnp.ndarray]:
        """Wraps the envpool reset() by converting arrays from numpy to JAX numpy.

        Returns:
            tuple[jnp.ndarray, jnp.ndarray]: Observations and infos for each env.
        """
        return jax.tree_util.tree_map(numpy_to_jax, self._env.reset())

    def _step(self, action: Any, env_id: int | None = None) -> tuple:
        """Wraps the envpool step() by converting arrays from numpy to JAX numpy.

        Args:
            action (Any): Action to take.
            env_id (int | None, optional): Internal env ID for envpool. Defaults to None.

        Returns:
            tuple: Step result.
        """
        result = self._env.step(action=action, env_id=env_id)
        return jax.tree_util.tree_map(numpy_to_jax, result)

    @functools.partial(jax.jit, static_argnums=0)
    def reset(self, _):
        # io_callback is required to handle random seeds correctly
        obs, info = jax.experimental.io_callback(self._reset, self.reset_shape)
        lives = jnp.array(info["lives"]) if self.is_atari else None

        return (self._handle0, lives), obs

    @functools.partial(jax.jit, static_argnums=0)
    def step(self, env_state: Any, action: Any, _):
        if not self.is_atari:
            env_state, _ = env_state
        else:
            env_state, lives = env_state

        # Here, we perform the actual step in the envpool environment
        env_state, (obs, reward, term, trunc, info) = self._xla_step(env_state, action)
        done = jnp.logical_or(term, trunc)

        # Since the envpool AutoReset differs from the one implemented in Gymnasium
        # we need this workaround to achieve the same behaviour.
        # To reset only certain environments that are done already
        # we need the non-jittable reset function
        def reset(obs, info):
            def reset_idx(i, obs):
                new_obs, _, _, _, _ = jax.experimental.io_callback(
                    self._step,
                    self.single_step_shape,
                    action=self.dummy_action[:1],
                    env_id=np.array([i]),
                )
                return obs

            for i in range(self._n_envs):
                if self.is_atari and self.episodic_life and self.has_lives:
                    obs = jax.lax.cond(
                        done[i] & (info["lives"][i] == 0),
                        lambda obs: reset_idx(i, obs),
                        lambda obs: obs,
                        obs,
                    )
                elif self.is_atari and not self.episodic_life and self.has_lives:
                    obs = jax.lax.cond(
                        lives_gone[i],
                        lambda obs: reset_idx(i, obs),
                        lambda obs: obs,
                        obs,
                    )
                else:
                    obs = jax.lax.cond(
                        done[i],
                        lambda obs: reset_idx(i, obs),
                        lambda obs: obs,
                        obs,
                    )

            return obs

        if self.is_atari:
            new_lives = jnp.array(info["lives"])
            lives_gone = new_lives < lives
        else:
            new_lives = None

        if self.is_atari and not self.episodic_life and self.has_lives:
            auto_reset = lives_gone
        else:
            auto_reset = done
        obs = jax.lax.cond(jnp.any(auto_reset), reset, lambda obs, info: obs, obs, info)

        return (env_state, new_lives), (obs, reward, done, info)

    @property
    def action_space(self):
        return gymnasium_space_to_gymnax_space(self._env.action_space)

    @property
    def observation_space(self):
        return gymnasium_space_to_gymnax_space(self._env.observation_space)
