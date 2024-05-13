import gymnasium as gym

env = gym.make('ALE/DoubleDunk-v5', render_mode="human")  # remove render_mode in training
obs, info = env.reset()
episode_over = False
while not episode_over:
    obs, reward, terminated, truncated, info = env.step(3)
    obs, reward, terminated, truncated, info = env.step(10)

    episode_over = terminated or truncated
env.close()