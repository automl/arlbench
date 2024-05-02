import envpool
env = envpool.make("Pong-v5", env_type="gymnasium", num_envs=1, seed=42, episodic_life=True)
for _ in range(1000):
    obs, info = env.reset()
    print(info["lives"])
    assert info["lives"] == 4, f"info['lives'] is {info['lives']}"
