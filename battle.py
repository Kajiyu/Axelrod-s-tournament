import numpy as np
import os, sys
import random

from utils import *


class Battle:
    def __init__(self, agent1, agent2, len_game=10000):
        self.agent1 = agent1
        self.agent2 = agent2
        self.game_history = []
        self.len_game = len_game
        self.score_1 = 0
        self.score_2 = 0
    
    def operate(self, debug=False):
        self.agent1.reset()
        self.agent2.reset()
        for t in range(self.len_game):
            a1 = self.agent1.act()
            a2 = self.agent2.act()
            # print(a1,a2)
            rwd_1 = RWD_MAP[a1, a2]
            rwd_2 = RWD_MAP[a2, a1]
            self.agent1.update([a1, a2], rwd_1, max_rwd=np.max(RWD_MAP))
            self.agent2.update([a2, a1], rwd_2, max_rwd=np.max(RWD_MAP))
            if debug:
                self.game_history.append([a1, a2])
            self.score_1 = self.score_1 + rwd_1
            self.score_2 = self.score_2 + rwd_2
        return [self.score_1, self.score_2]