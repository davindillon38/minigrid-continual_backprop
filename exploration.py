import numpy as np
from collections import defaultdict

class CountBasedExploration:
    def __init__(self, bonus_coef=0.01):
        self.counts = defaultdict(int)
        self.bonus_coef = bonus_coef
    
    def hash_state(self, obs):
        # Hash the image observation - convert tensor to numpy
        return obs['image'].cpu().numpy().tobytes()
    
    def get_bonus(self, obs):
        state_hash = self.hash_state(obs)
        self.counts[state_hash] += 1
        count = self.counts[state_hash]
        return self.bonus_coef / np.sqrt(count)