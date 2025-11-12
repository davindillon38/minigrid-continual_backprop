import gymnasium as gym
import numpy as np
from collections import deque
from PIL import Image

class AtariPreprocessing(gym.Wrapper):
    def __init__(self, env, frame_stack=4):
        super().__init__(env)
        self.frame_stack = frame_stack
        self.frames = deque(maxlen=frame_stack)
        self.observation_space = gym.spaces.Box(
                low=0, high=1, shape=(frame_stack, 84, 84), dtype=np.float32
            )
        
    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        obs = self._preprocess(obs)
        for _ in range(self.frame_stack):
            self.frames.append(obs)
        return self._get_obs(), info
    
    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        obs = self._preprocess(obs)
        self.frames.append(obs)
        return self._get_obs(), reward, terminated, truncated, info
    
    def _preprocess(self, frame):
        # RGB to grayscale, resize to 84x84
        gray = np.dot(frame[...,:3], [0.299, 0.297, 0.114])
        resized = np.array(Image.fromarray(gray.astype(np.uint8)).resize((84, 84)))
        return resized.astype(np.float32) / 255.0
    
    def _get_obs(self):
        return np.stack(self.frames, axis=0)  # (4, 84, 84)