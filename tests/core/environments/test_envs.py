import jax
import time
import jax.numpy as jnp

from arlbench.core.algorithms import DQN

from arlbench.core.environments import make_env

def test_gymnasium_env():
    env = make_env("gymnasium", "CartPole-v1", n_envs=10, seed=42)

    rng = jax.random.PRNGKey(42)
    env_state, obs = env.reset(rng)
    action = env.sample_actions(rng)

    env_state, (obs, reward, done, info) = env.step(env_state, action, rng)
    print((obs, reward, done, info))

if __name__ == "__main__":
    test_gymnasium_env()
    
