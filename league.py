import numpy as np
import os, sys
import random

from utils import *
from battle import Battle


class League:
    def __init__(self, agents_dict):
        self.agents = agents_dict
        self.agent_keys = list(agents_dict.keys())
        self.n_agents = len(self.agent_keys)
        self.score_matrix = np.ones((self.n_agents, self.n_agents))*(-1)
    
    def operate(self, len_game=10000, p_debug=False):
        for i in range(self.n_agents):
            for j in range(self.n_agents):
                if i == j:
                    self.score_matrix[i, i] = 0
                    continue
                elif self.score_matrix[i, j] < 0 and self.score_matrix[j, i] < 0:
                    agent1 = self.agents[self.agent_keys[i]]
                    agent2 = self.agents[self.agent_keys[j]]
                    battle = Battle(agent1, agent2, len_game=len_game)
                    scores = battle.operate()
                    self.score_matrix[i, j] = scores[0]
                    self.score_matrix[j, i] = scores[1]
                    if p_debug:
                        print(self.agent_keys[i], "vs", self.agent_keys[j], ":", scores)
                else:
                    continue