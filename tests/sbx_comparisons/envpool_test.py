import envpool
import jax
import jax.numpy as jnp
import numpy as np
import cv2

env = envpool.make("Breakout-v5", env_type="gym", num_envs=8, seed=42, episodic_life=False)
handle1, _, _, xla_step = env.xla()

action = jnp.array([0, 0, 0, 0, 0, 0, 0, 0])
np_action = np.array([0, 0, 0, 0, 0, 0, 0, 0])
output = env.reset()

reset_shape = jax.tree_map(lambda x: x[:1], output)

step_shape = env.step(np_action)
single_step_shape = jax.tree_map(lambda x: x[:1], step_shape)
output = env.reset()


img_list = []
for i in range(27000):
    handle1, (obs1, reward1, done1, trunc1, info1) = xla_step(handle1, action)
    # store obs as image
    np_obs = obs1.numpy()
    scaled_obs = (255 * np_obs).astype(np.uint8)  # Scale to 0-255 as uint8
    img_list.append(scaled_obs)
    if done1:  # Check if the environment says the episode is complete
        print("early stopping")
        break

video_filename = 'output_video.mp4'
height, width = img_list[0].shape
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Using DIVX codec
video = cv2.VideoWriter(video_filename, fourcc, 30, (width, height), isColor=False)
# Write each frame to the video
for img in img_list:
    video.write(img)

# Release the video writer
video.release()


handle1, (obs1, reward1, done1, trunc1, info1) = xla_step(handle1, action)
obs2, reward2, done2, trunc2, info2 = jax.experimental.io_callback(env.step, single_step_shape, np_action[:1], env_id=np.array([0]))
#obs1, reward1, done1, trunc1, info1 = env.step(np_action)
handle3, (obs3, reward3, done3, trunc3, info3) = xla_step(handle1, action)
#obs2, reward2, done2, trunc2, info2 = env.step(np_action[:1], np.array([0]))

a = 0


