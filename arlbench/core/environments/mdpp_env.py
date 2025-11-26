"""Gymnax Version of MDP Playground's RLToyEnv. Only discrete envs for now."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import warnings
import logging
import functools
from typing import TYPE_CHECKING, Any

import jax
from jax import numpy as jnp
from flax import struct
from gymnax.environments import environment, spaces
from torch import ne
from arlbench.core.environments.autorl_env import Environment
from omegaconf import OmegaConf 

if TYPE_CHECKING:
    from chex import PRNGKey


class DiscreteExtendedSampling(spaces.Discrete):
    """Minimal jittable class for discrete gymnax spaces with different sampling."""
    def sample(self, key: jax.Array, max=None, prob=None, replace=True) -> jax.Array:
        """Sample random action uniformly from set of categorical choices."""
        if max is None:
            max = self.n
        return jax.random.choice(key, max, p=prob, replace=replace)


@struct.dataclass
class MDPPState(environment.EnvState):
    curr_state: jax.Array
    augmented_state: jax.Array
    reward: jax.Array
    reward_buffer: jax.Array
    total_abs_noise_in_reward_episode: float
    total_abs_noise_in_transition_episode: float
    total_noisy_transitions_episode: int
    total_reward_episode: float
    total_transitions_episode: int
    reached_terminal: bool = False


@struct.dataclass
class EnvParams(environment.EnvParams):
    """Environment parameters for ToyEnv."""
    delay: int = 0  # Delays each reward by this number of timesteps. Default value: 0.
    sequence_length: int = 1  # Intrinsic sequence length of the reward function of an environment. For discrete environments, randomly selected sequences of this length are set to be rewardable at initialisation if use_custom_mdp = false and generate_random_mdp = true. Default value: 1.
    transition_noise = 0  # float in range [0, 1] or Python function(state, action, rng)
    reward_noise = 0  # float or Python function(state, action, rng)
    reward_density: float = 0.25  # float in range [0, 1]
    reward_scale: float = 1.0  # Multiplies the rewards by this value at every time step. Default value: 1.
    reward_shift: float = (
        0.0  # This value is added to the reward at every time step. Default value: 0.
    )
    diameter: int = 1  # For discrete environments, if diameter = d, the set of states is set to be a d-partite graph (and NOT a complete d-partite graph), where, if we order the d sets as 1, 2, .., d, states from set 1 will have actions leading to states in set 2 and so on, with the final set d having actions leading to states in set 1. Number of actions for each state will, thus, be = (number of states) / (d). Default value: 1 for discrete environments. For continuous environments, this dimension is set automatically based on the state_space_max value.
    terminal_state_density: float = 0.25  # float in range [0, 1]
    term_state_reward: float = 0
    use_custom_mdp: bool = False  # If true, users specify their own transition and reward functions using the config options transition_function and reward_function (see below). Optionally, they can also use init_state_dist and terminal_states for discrete spaces (see below).
    transition_function = None  # Python function(state, action) or a 2-D numpy.ndarray
    reward_function = None
    make_denser: bool = False  # If true, makes the reward denser in environments
    action_space_size: int = 4  # Size of action space for discrete environments
    repeats_in_sequences: bool = True  # Whether to allow repeats in rewardable sequences for discrete environments
    initial_state_dist: jax.Array | None = None  # Initial state distribution, rho_0, for discrete environments
    transition_function: jax.Array | None = None  # Transition function matrix, P, for discrete environments
    reward_function: dict | None = None  # Reward function, R, for discrete
    maximally_connected: bool = False  # If true, makes every state able to transition to every other state in the next independent set for discrete environments
    reward_every_n_steps: int = 1  # Rewards are given only every n steps if this is > 1
    terminal_states: jax.Array | None = None  # Set of terminal states for discrete environments
    initial_reward_buffer: jax.Array = None  # Initial reward buffer for delayed rewards
    rewardable_sequences: jax.Array | None = None  # Set of rewardable sequences for discrete environments
    max_steps_in_episode: int | None = 100  # Maximum steps in an episode


class RLToyEnv(environment.Environment):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}

    """
    The base toy environment in MDP Playground. It is parameterised by a config dict and can be instantiated to be an MDP with any of the possible dimensions from the accompanying research paper. The class extends OpenAI Gym's environment gym.Env.

    The accompanying paper is available at: https://arxiv.org/abs/1909.07750.

    This Jax version is adapted from the original implementation available and only implements a part of the discrete options.
    """

    def action_space(self, params: EnvParams | None = None) -> spaces.Discrete:
        """Action space of the environment."""
        return DiscreteExtendedSampling(params.action_space_size)

    def observation_space(self, params: EnvParams) -> spaces.Box:
        """Observation space of the environment."""
        state_space_size = params.action_space_size * params.diameter
        return DiscreteExtendedSampling(state_space_size)

    def transition_function(self, params, state, action):
        """The transition function, P.

        Parameters
        ----------
        state : list
            The state that the environment will use to perform a transition.
        action : list
            The action that the environment will use to perform a transition.

        Returns
        -------
        int or np.array
            The state at the end of the current transition
        """
        next_state = params.transition_function[state, action]

        noisy_transitions = 0
        absolute_transition_noise = 0.0

        if params.transition_noise:
            probs = (
                jnp.ones(shape=(params.state_space_size[0],))
                * params.transition_noise
                / (params.state_space_size[0] - 1)
            )
            probs[next_state] = 1 - params.transition_noise
            new_next_state = self.observation_spaces[0].sample(prob=probs)  # random
            absolute_transition_noise = jnp.abs(new_next_state - next_state)
            if next_state != new_next_state:
                noisy_transitions += 1
            next_state = new_next_state

        next_state = jnp.array(next_state).astype(int)

        return next_state, jnp.array(noisy_transitions), jnp.array(absolute_transition_noise)

    def reward_function(self, params, state_considered, reward_buffer, total_transitions_episode, key):
        """The reward function, R.

        Rewards the sequences selected to be rewardable at initialisation for discrete environments. For continuous environments, we have fixed available options for the reward function:
            move_to_a_point rewards for moving to a predefined location. It has sparse and dense settings.
            move_along_a_line rewards moving along ANY direction in space as long as it's a fixed direction for sequence_length consecutive steps.

        Parameters
        ----------
        state : list
            The underlying MDP state (also called augmented state in this code) that the environment uses to calculate its reward. Normally, just the sequence of past states of length delay + sequence_length + 1.
        action : single action dependent on action space
            Action magnitudes are penalised immediately in the case of continuous spaces and, in effect, play no role for discrete spaces as the reward in that case only depends on sequences of states. We say "in effect" because it _is_ used in case of a custom R to calculate R(s, a) but that is equivalent to using the "next" state s' as the reward determining criterion in case of deterministic transitions. _Sequences_ of _actions_ are currently NOT used to calculate the reward. Since the underlying MDP dynamics are deterministic, a state and action map 1-to-1 with the next state and so, just a sequence of _states_ should be enough to calculate the reward.

        Returns
        -------
        double
            The reward at the end of the current transition

        """
        delayed_seq = jax.lax.dynamic_slice(jnp.array(state_considered), (1+params.delay,), (self.augmented_state_length-1,))
        sequence_matches = jnp.all(params.rewardable_sequences.astype(int) == delayed_seq.astype(int), axis=1)
        reward = jnp.sum(sequence_matches * params.reward_function)

        reward_buffer = jnp.concatenate((jnp.array(reward_buffer), jnp.array([reward])))
        reward = reward_buffer[0]
        reward_buffer = reward_buffer[1:]
        
        should_zero = (total_transitions_episode % params.reward_every_n_steps) != 0
        reward = jnp.where(should_zero, 0.0, reward)

        if params.reward_noise:
            noise_in_reward = jax.random.generalized_normal(key, shape=(1,), p=params.reward_noise)
        else:
            noise_in_reward = 0.0

        raw_reward = reward
        reward += noise_in_reward
        reward *= params.reward_scale
        reward += params.reward_shift
        return raw_reward, reward, reward_buffer, noise_in_reward

    def step_env(
        self,
        key: jax.Array,
        state: MDPPState,
        action: int | float | jax.Array,
        params: EnvParams,
    ) -> tuple[jax.Array, MDPPState, jax.Array, jax.Array, dict[Any, Any]]:
        """Perform single timestep state transition."""

        curr_state = state.curr_state
        next_state, noisy_transitions_episode, total_abs_noise_in_transition_episode = self.transition_function(params, curr_state, action)
        total_noisy_transitions_episode = state.total_noisy_transitions_episode + noisy_transitions_episode
        augmented_state = state.augmented_state

        augmented_state = augmented_state[1:]
        augmented_state = jnp.concatenate((jnp.array(augmented_state), jnp.array(next_state)))

        total_transitions_episode = state.total_transitions_episode + 1

        raw_reward, reward, reward_buffer, reward_noise = self.reward_function(params, augmented_state, state.reward_buffer, total_transitions_episode, key)
        total_reward_episode = state.total_reward_episode + raw_reward
        total_abs_noise_in_reward_episode = state.total_abs_noise_in_reward_episode + jnp.abs(reward_noise)

        next_obs = next_state
        curr_obs = jnp.array(next_obs)

        terminal_states = jnp.array(params.terminal_states) 
        matches = jnp.all(terminal_states == next_state, axis=-1)  # shape (N,)
        done = jnp.any(matches)
        done = jnp.logical_or(done, total_transitions_episode > params.max_steps_in_episode)

        def add_terminal_rewards():
            return reward + params.term_state_reward * params.reward_scale
        
        def return_reward():
            return reward
        
        reward = jax.lax.cond(done, add_terminal_rewards, return_reward)

        next_state = MDPPState( 
            curr_state=jnp.array(next_state),
            augmented_state=augmented_state,
            reward=reward,
            reward_buffer=reward_buffer,
            reached_terminal=done,
            total_abs_noise_in_reward_episode=total_abs_noise_in_reward_episode,
            total_abs_noise_in_transition_episode=total_abs_noise_in_transition_episode,
            total_noisy_transitions_episode=total_noisy_transitions_episode,
            total_reward_episode=total_reward_episode,
            total_transitions_episode=total_transitions_episode,
            time=state.time + 1,
        )

        return (
            jax.lax.stop_gradient(curr_obs),
            next_state,
            reward,
            done,
            {
                "curr_state": state.curr_state,
                "curr_obs": curr_obs,
                "augmented_state": state.augmented_state,
            },
        )

    def reset_env(
        self, key: jax.Array, params: EnvParams
    ) -> tuple[jax.Array, MDPPState]:
        """Resets the environment for the beginning of an episode and samples a start state from rho_0. For discrete environments uses the defined rho_0 directly. For continuous environments, samples a state and resamples until a non-terminal state is sampled.

        Returns
        -------
        int or np.array
            The start state for a new episode.
        """
        curr_state = jax.random.categorical(key, logits=jnp.log(params.initial_state_dist))
        augmented_state = [jnp.nan for _ in range(self.augmented_state_length - 1)]
        augmented_state.append(curr_state)
        curr_state = jnp.array([curr_state])

        state = MDPPState(
            curr_state=curr_state,
            augmented_state=jnp.array(augmented_state),
            reward=jnp.array(0.0),
            reward_buffer=params.initial_reward_buffer,
            reached_terminal=jnp.array(False),
            total_abs_noise_in_reward_episode=jnp.array(0.0),
            total_abs_noise_in_transition_episode=jnp.array(0.0),
            total_noisy_transitions_episode=jnp.array(0),
            total_reward_episode=jnp.array(0.0),
            total_transitions_episode=jnp.array(0),
            time=jnp.array(0)
        )

        return curr_state, state


def dist_of_pt_from_line(pt, ptA, ptB):
    """Returns shortest distance of a point from a line defined by 2 points - ptA and ptB.
    Based on: https://softwareengineering.stackexchange.com/questions/168572/distance-from-point-to-n-dimensional-line"""

    tolerance = 1e-13
    lineAB = ptA - ptB
    lineApt = ptA - pt
    dot_product = jnp.dot(lineAB, lineApt)
    if jnp.linalg.norm(lineAB) < tolerance:
        return 0
    else:
        proj = dot_product / jnp.linalg.norm(lineAB)
        sq_dist = jnp.linalg.norm(lineApt) ** 2 - proj**2

        if sq_dist < 0:
            if sq_dist < -tolerance:
                logging.warning(
                    "The squared distance calculated in dist_of_pt_from_line()"
                    " using Pythagoras' theorem was less than the tolerance allowed."
                    " It was: " + str(sq_dist) + ". Tolerance was: -" + str(tolerance)
                )
            sq_dist = 0
        dist = jnp.sqrt(sq_dist)
        return dist
    
def init_terminal_states(params):
        """Initialises terminal state set to be the 'last' states for discrete environments. For continuous environments, terminal states will be in a hypercube centred around config['terminal_states'] with the edge of the hypercube of length config['term_state_edge']."""
        # Define the no. of terminal states per independent set of the state space
        num_terminal_states = params["terminal_state_density"] * params["action_space_size"]
        terminal_states = jnp.array(
                [
                    j * params["action_space_size"] - 1 - i
                    for j in range(1, int(params["diameter"]) + 1)
                    for i in range(int(num_terminal_states))
                ]
            )  # terminal states

        return terminal_states
    

def get_init_state_dist(params):
        """Initialises initial state distrbution, rho_0, to be uniform over the non-terminal states for discrete environments. For both discrete and continuous environments, the uniform sampling over non-terminal states is taken care of in reset() when setting the initial state for an episode."""
        non_term_state_space_size = int(
                params["action_space_size"] - params["terminal_state_density"] * params["action_space_size"]
            )  # #hardcoded
        init_state_dist = (
                [
                    1 / (non_term_state_space_size * params["diameter"])
                    for i in range(non_term_state_space_size)
                ]
                + [0 for i in range(int(params["terminal_state_density"] * params["action_space_size"]))]
            ) * int(params["diameter"])
        return jnp.array(init_state_dist)

def init_transition_function(params, key):
        """Initialises transition function, P by selecting random next states for every (state, action) tuple for discrete environments. For continuous environments, we have 1 option for the transition function which varies depending on dynamics order and inertia and time_unit for a point object."""
        # relevant dimensions part
        state_space_size = params["action_space_size"] * int(params["diameter"])
        dummy_obs_space = DiscreteExtendedSampling(state_space_size)
        transition_function = jnp.ones(shape=(state_space_size, params["action_space_size"]),) * -1
        if params["maximally_connected"] and params["diameter"] == 1:
            for s in range(state_space_size):
                transition_function = transition_function.at[s].set(dummy_obs_space.sample(size=params["action_space_size"], replace=False, key=key))
        else: 
            for s in range(state_space_size):
                i_s = s // params["action_space_size"]  # select the current independent

                # Set the probabilities of the next state for the current independent set
                prob = jnp.zeros(shape=(state_space_size,))
                prob_next_states = (
                    jnp.ones(shape=(params["action_space_size"],))
                    / params["action_space_size"]
                )
                ind_1 = ((i_s + 1) * params["action_space_size"]) % state_space_size
                ind_2 = ((i_s + 2) * params["action_space_size"]) % state_space_size
                if ind_2 <= ind_1:  # edge case
                    ind_2 += state_space_size
                prob = prob.at[ind_1:ind_2].set(prob_next_states)

                if params["maximally_connected"]:
                    transition_function = transition_function.at[s].set(dummy_obs_space.sample(prob=prob, size=params["action_space_size"], replace=False, key=key))
                else:
                    for a in range(params["action_space_size"]):
                        sample_key, key = jax.random.split(key)
                        transition_function = transition_function.at[s, a].set(dummy_obs_space.sample(prob=prob, key=sample_key).item())
                
        # Set the next state for terminal states to be themselves, for any action taken.
        for i_s in range(int(params["diameter"])):
            for s in range(
                int(params["action_space_size"] - params["terminal_state_density"] * params["action_space_size"]),
                int(params["action_space_size"]),
            ):
                for a in range(params["action_space_size"]):
                    transition_function = transition_function.at[
                        i_s * params["action_space_size"] + s, a
                    ].set(i_s * params["action_space_size"] + s)
        transition_function = transition_function.astype(int)

        return transition_function

def init_reward_function(params, key):
        """Initialises reward function, R by selecting random sequences to be rewardable for discrete environments. For continuous environments, we have fixed available options for the reward function."""

        non_term_state_space_size = params["action_space_size"] - params["terminal_state_density"] * params["action_space_size"]

        def get_sequences(maximum, length, fraction, repeats=False, diameter=1):
            """
            Returns random sequences of integers

            maximum: int
                Max value of the integers in the sequence
            length: int
                Length of sequence
            fraction: float
                Fraction of total possible sequences to be returned
            repeats: boolean
                Allows repeats in returned sequences
            diameter: int
                Relates to the diameter of the MDP
            """

            sequences = []

            if repeats:
                num_possible_sequences = (maximum) ** length
                num_sel_sequences = int(fraction * num_possible_sequences)
                if num_sel_sequences == 0:
                    num_sel_sequences = 1
                    warnings.warn(
                        "0 rewardable sequences per independent"
                        " set for given reward_density, sequence_length,"
                        " diameter and terminal_state_density. Setting it to 1."
                    )

                sel_sequence_nums = jax.random.choice(
                    key,
                    num_possible_sequences,
                    shape=(num_sel_sequences,),
                    replace=False,
                )
                for i_s in range(int(diameter)):
                    for i in range(num_sel_sequences):
                        curr_sequence_num = sel_sequence_nums[i]
                        specific_sequence = []
                        while len(specific_sequence) != length:
                            specific_sequence.append((
                                curr_sequence_num % (non_term_state_space_size)
                                + ((len(specific_sequence) + i_s) % diameter)
                                * params["action_space_size"]
                            ).item())
                            curr_sequence_num = curr_sequence_num // (
                                non_term_state_space_size
                            )
                        sequences.append(specific_sequence)
            else:  # if no repeats
                assert length <= diameter * maximum, (
                    "When there are no"
                    " repeats in sequences, the sequence length should be"
                    " <= diameter * maximum."
                )
                permutations = []
                for i in range(length):
                    permutations.append(maximum - (i // diameter))

                for i_s in range(diameter):
                    num_possible_permutations = jnp.prod(permutations)
                    num_sel_sequences = int(fraction * num_possible_permutations)
                    if num_sel_sequences == 0:
                        num_sel_sequences = 1
                        warnings.warn(
                            "0 rewardable sequences per"
                            " independent set for given reward_density,"
                            " sequence_length, diameter and"
                            " terminal_state_density. Setting it to 1."
                        )

                    sel_sequence_nums = jax.random.choice(
                        key,
                        num_possible_permutations,
                        size=num_sel_sequences,
                        replace=False,
                    )

                    total_clashes = 0
                    for i in range(num_sel_sequences):
                        curr_permutation = sel_sequence_nums[i]
                        seq_ = []
                        curr_rem_digits = []
                        for j in range(diameter):
                            curr_rem_digits.append(list(range(maximum)))
                        for enum, j in enumerate(permutations):  # Goes
                            # from largest to smallest number among the factors of nPk
                            rem_ = curr_permutation % j
                            # rem_ = (enum // maximum) * maximum + rem_
                            seq_.append(
                                curr_rem_digits[(enum + i_s) % diameter][rem_]
                                + (
                                    ((enum + i_s) % diameter)
                                    * params.action_space_size
                                )
                            )
                            del curr_rem_digits[(enum + i_s) % diameter][rem_]
                            curr_permutation = curr_permutation // j

                        if seq_ in sequences:  # #hack
                            total_clashes += 1  
                        sequences.append(seq_.item())

                    assert total_clashes == 0, (
                        "None of the generated"
                        " sequences should have clashed with an existing"
                        " rewardable sequence when it was generated. No. of"
                        " times a clash was detected:" + str(total_clashes)
                    )
            return sequences

        rewardable_sequences = {}
        def insert_sequence(sequence):
            """
            Inserts rewardable sequences into the rewardable_sequences dict member variable
            """
            sequence = tuple(sequence)
            rewardable_sequences[sequence] = 1.0

            if params["make_denser"]:
                for ss_len in range(1, len(sequence)):
                    sub_sequence = tuple(sequence[:ss_len])
                    if sub_sequence not in rewardable_sequences:
                        rewardable_sequences[sub_sequence] = 0.0
                    rewardable_sequences[sub_sequence] += (
                        rewardable_sequences[sequence] * ss_len / len(sequence)
                    )

        sequences = get_sequences(
                maximum=non_term_state_space_size,
                length=params["sequence_length"],
                fraction=params["reward_density"],
                repeats=params["repeats_in_sequences"],
                diameter=params["diameter"],
            )

        if len(sequences) > 1000:
            warnings.warn(
                "Too many rewardable sequences and/or too long"
                " rewardable sequence length. Environment might be too slow."
                " Please consider setting the reward_density to be lower or"
                " reducing the sequence length. No. of rewardable sequences:"
                + str(len(sequences))
            )

        for specific_sequence in sequences:
            insert_sequence(specific_sequence)

        seq_keys = list(rewardable_sequences.keys())
        seq_keys = jnp.array([jnp.array(list(k)).astype(int) for k in seq_keys]).astype(int)
        seq_values = jnp.array(list(rewardable_sequences.values()))
        return jnp.array(seq_keys), jnp.array(seq_values)


def list_to_float_np_array(lis):
    """Converts list to numpy float array"""
    return jnp.array(list(float(i) for i in lis))


class JaxMdppEnv(Environment):
    """A small jax version of the MDPP environment."""

    def __init__(
        self, env_name: str, n_envs: int, env_kwargs: dict[str, Any] | None = None
    ):
        """Creates a gymnax environment for JAX-based RL training.

        Args:
            env_name (str): Name/id of the brax environment.
            n_envs (int): Number of environments.
            env_kwargs (dict[str, Any] | None, optional): Keyword arguments
                to pass to the gymnax environment. Defaults to None.
        """
        if env_kwargs is None:
            env_kwargs = {}
        if not isinstance(env_kwargs, dict):
            env_kwargs = OmegaConf.to_container(env_kwargs)
        
        if "action_space_size" not in env_kwargs:
            env_kwargs["action_space_size"] = 4
        if "diameter" not in env_kwargs:
            env_kwargs["diameter"] = 1
        if "terminal_state_density" not in env_kwargs:
            env_kwargs["terminal_state_density"] = 0.25
        if "maximally_connected" not in env_kwargs:
            env_kwargs["maximally_connected"] = False
        if "sequence_length" not in env_kwargs:
            env_kwargs["sequence_length"] = 1
        if "reward_density" not in env_kwargs:
            env_kwargs["reward_density"] = 0.25
        if "repeats_in_sequences" not in env_kwargs:
            env_kwargs["repeats_in_sequences"] = True
        if "make_denser" not in env_kwargs:
            env_kwargs["make_denser"] = False     

        env = RLToyEnv()
        env_kwargs["initial_state_dist"] = get_init_state_dist(env_kwargs)
        rng_key = jax.random.PRNGKey(0)
        env_kwargs["transition_function"] = init_transition_function(env_kwargs, rng_key)
        env_kwargs["rewardable_sequences"], env_kwargs["reward_function"] = init_reward_function(env_kwargs, rng_key)
        env_kwargs["terminal_states"] = init_terminal_states(env_kwargs)
        env_kwargs["initial_reward_buffer"] = jnp.zeros(shape=(env_kwargs.get("delay", 0),))
        env_params = EnvParams(**env_kwargs)
        super().__init__(env_name, env, n_envs)

        self.env_params = env_params
        self._env.augmented_state_length = env_params.sequence_length + env_params.delay + 1

    @functools.partial(jax.jit, static_argnums=0)
    def reset(self, rng: jax.random.PRNGKey):
        """Resets the environment."""
        reset_rng = jax.random.split(rng, self.n_envs)
        obs, env_state = jax.vmap(self._env.reset, in_axes=(0, None))(
            reset_rng, self.env_params
        )
        return env_state, obs

    @functools.partial(jax.jit, static_argnums=0)
    def step(self, env_state: Any, action: Any, rng: jax.random.PRNGKey):
        """Steps the environment forward."""
        step_rng = jax.random.split(rng, self.n_envs)
        obs, env_state, reward, done, info = jax.vmap(
            self._env.step, in_axes=(0, 0, 0, None)
        )(step_rng, env_state, action, self.env_params)
        self.sample_action(rng)
        return env_state, (obs, reward, done, info)

    @property
    def action_space(self):
        """Action space of the environment."""
        return self._env.action_space(self.env_params)

    @functools.partial(jax.jit, static_argnums=0)
    def sample_action(self, rng: jax.random.PRNGKey):
        """Samples a random action from the action space."""
        actions = self.action_space.sample(rng)
        return actions

    @property
    def observation_space(self):
        """Observation space of the environment."""
        return self._env.observation_space(self.env_params)