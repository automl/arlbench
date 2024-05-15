from __future__ import annotations

import gymnasium


class ImageExtractionWrapper(gymnasium.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self.observation_space = self.env.observation_space["image"]

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        return obs["image"], info

    def step(self, action):
        obs, reward, tr, te, info = self.env.step(action)
        return obs["image"], reward, tr, te, info
