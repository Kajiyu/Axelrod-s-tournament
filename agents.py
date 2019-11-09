import numpy as np
import os, sys
import random

from utils import *


class AgentBase:
    def __init__(self, memory_size):
        self.memory_size = memory_size
        self.memory_list = []
        self.Q = np.zeros((4**self.memory_size, 2))
    
    def update(self, actions, reward, max_rwd=5.0):
        raise NotImplementedError()
    
    def act(self):
        raise NotImplementedError()


class All_C_Agent(AgentBase):
    def __init__(self, memory_size=0):
        super().__init__(memory_size)
    
    def update(self, actions, reward, max_rwd=5.0):
        pass
    
    def act(self):
        return 0
    
    def reset(self):
        self.memory_list = []


class All_D_Agent(AgentBase):
    def __init__(self, memory_size=0):
        super().__init__(memory_size)
    
    def update(self, actions, reward, max_rwd=5.0):
        pass
    
    def act(self):
        return 1
    
    def reset(self):
        self.memory_list = []


class Extort_2_Agent(AgentBase):
    def __init__(self, memory_size=1):
        super().__init__(memory_size)
        self.Q[0,0] = 8./9.
        self.Q[0,1] = 1./9.
        self.Q[1,0] = 1./2.
        self.Q[1,1] = 1./2.
        self.Q[2,0] = 1./3.
        self.Q[2,1] = 2./3.
        self.Q[3,0] = 0.
        self.Q[3,1] = 1.
    
    def update(self, actions, reward, max_rwd=5.0):
        self.memory_list.append(actions)
        if len(self.memory_list) > self.memory_size:
            self.memory_list = self.memory_list[(-1)*self.memory_size:]
    
    def act(self):
        if len(self.memory_list) < self.memory_size:
            return round(random.random())
        state = int(list_to_str(np.array(self.memory_list).reshape(-1)), 2)
        qs = self.Q[state, :].reshape(-1)
        action = np.random.choice([0, 1], p=qs)
        return action
    
    def reset(self):
        self.memory_list = []


class Hard_Majo_Agent(AgentBase):
    def __init__(self, memory_size=1):
        super().__init__(memory_size)
        self.num_c = 0
        self.num_d = 0
    
    def update(self, actions, reward, max_rwd=5.0):
        if actions[1] == 0:
            self.num_c = self.num_c + 1
        else:
            self.num_d = self.num_d + 1
    
    def act(self):
        if self.num_c + self.num_d < 1:
            return 1
        else:
            if self.num_c > self.num_d:
                return 0
            else:
                return 1
    
    def reset(self):
        self.memory_list = []


class Hard_Joss_Agent(AgentBase):
    def __init__(self, memory_size=1):
        super().__init__(memory_size)
        self.Q[0,0] = 0.9
        self.Q[0,1] = 0.1
        self.Q[1,0] = 0.
        self.Q[1,1] = 1.
        self.Q[2,0] = 1.
        self.Q[2,1] = 0.
        self.Q[3,0] = 0.
        self.Q[3,1] = 1.
    
    def update(self, actions, reward, max_rwd=5.0):
        self.memory_list.append(actions)
        self.memory_list = self.memory_list[(-1)*self.memory_size:]
    
    def act(self):
        if len(self.memory_list) < self.memory_size:
            return round(random.random())
        state = int(list_to_str(np.array(self.memory_list).reshape(-1)), 2)
        qs = self.Q[state, :].reshape(-1)
        action = np.random.choice([0, 1], p=qs)
        return action
    
    def reset(self):
        self.memory_list = []


class Hard_TFT_Agent(AgentBase):
    def __init__(self, memory_size=3):
        super().__init__(memory_size)
    
    def update(self, actions, reward, max_rwd=5.0):
        self.memory_list.append(actions)
        if len(self.memory_list) > self.memory_size:
            self.memory_list = self.memory_list[(-1)*self.memory_size:]
    
    def act(self):
        if len(self.memory_list) < self.memory_size:
            return round(random.random())
        if np.sum(np.array(self.memory_list)[:, 1]) > 0.:
            return 1
        else:
            return 0
    
    def reset(self):
        self.memory_list = []


class Hard_TF2T_Agent(AgentBase):
    def __init__(self, memory_size=3):
        super().__init__(memory_size)
    
    def update(self, actions, reward, max_rwd=5.0):
        self.memory_list.append(actions)
        if len(self.memory_list) > self.memory_size:
            self.memory_list = self.memory_list[(-1)*self.memory_size:]
    
    def act(self):
        if len(self.memory_list) < self.memory_size:
            return round(random.random())
        if np.array(self.memory_list)[1, 1] == 0:
            return 0
        else:
            if np.array(self.memory_list)[0, 1] == 1 or np.array(self.memory_list)[2, 1] == 1:
                return 1
            else:
                return 0
    
    def reset(self):
        self.memory_list = []


class Hard_TFT_Agent(AgentBase):
    def __init__(self, memory_size=1):
        super().__init__(memory_size)
    
    def update(self, actions, reward, max_rwd=5.0):
        self.memory_list.append(actions)
        if len(self.memory_list) > self.memory_size:
            self.memory_list = self.memory_list[(-1)*self.memory_size:]
    
    def act(self):
        if len(self.memory_list) < self.memory_size:
            return round(random.random())
        if np.array(self.memory_list)[:, 1] > 0.:
            return 1
        else:
            return 0
    
    def reset(self):
        self.memory_list = []


class TFT_Agent(AgentBase):
    def __init__(self, memory_size=1):
        super().__init__(memory_size)
    
    def update(self, actions, reward, max_rwd=5.0):
        self.memory_list.append(actions)
        if len(self.memory_list) > self.memory_size:
            self.memory_list = self.memory_list[(-1)*self.memory_size:]
    
    def act(self):
        if len(self.memory_list) == 0:
            return 0
        else:
            return np.array(self.memory_list)[-1, 1]
    
    def reset(self):
        self.memory_list = []


class Grim_Agent(AgentBase): # Trigger
    def __init__(self, memory_size=1):
        super().__init__(memory_size)
        self.mode = 0
    
    def update(self, actions, reward, max_rwd=5.0):
        if self.mode == 0:
            if actions[1] == 1:
                self.mode = 1
    
    def act(self):
        if self.mode == 0:
            return 0
        else:
            return 1
    
    def reset(self):
        self.memory_list = []


class GTFT_Agent(AgentBase):
    def __init__(self, memory_size=1):
        super().__init__(memory_size)
        self.Q[0,0] = 1.
        self.Q[0,1] = 0.
        self.Q[1,0] = 1./3.
        self.Q[1,1] = 2./3.
        self.Q[2,0] = 1.
        self.Q[2,1] = 0.
        self.Q[3,0] = 1./3.
        self.Q[3,1] = 2./3.
    
    def update(self, actions, reward, max_rwd=5.0):
        self.memory_list.append(actions)
        self.memory_list = self.memory_list[(-1)*self.memory_size:]
    
    def act(self):
        if len(self.memory_list) < self.memory_size:
            return round(random.random())
        state = int(list_to_str(np.array(self.memory_list).reshape(-1)), 2)
        qs = self.Q[state, :].reshape(-1)
        action = np.random.choice([0, 1], p=qs)
        return action
    
    def reset(self):
        self.memory_list = []


class TF2T_Agent(AgentBase):
    def __init__(self, memory_size=1):
        super().__init__(memory_size)
        self.mode = 0
    
    def update(self, actions, reward, max_rwd=5.0):
        if self.mode == 0:
            if actions[1] == 1:
                self.mode = 1
        elif self.mode == 1:
            if actions[1] == 1:
                self.mode = 2
        else:
            if actions[1] == 0:
                self.mode = 0
    
    def act(self):
        if self.mode == 2:
            return 1
        else:
            return 0
    
    def reset(self):
        self.memory_list = []


class WSLS_Agent(AgentBase):
    def __init__(self, memory_size=1):
        super().__init__(memory_size)
        self.Q[0,0] = 1.
        self.Q[0,1] = 0.
        self.Q[1,0] = 0.
        self.Q[1,1] = 1.
        self.Q[2,0] = 0.
        self.Q[2,1] = 1.
        self.Q[3,0] = 1.
        self.Q[3,1] = 0.
    
    def update(self, actions, reward, max_rwd=5.0):
        self.memory_list.append(actions)
        self.memory_list = self.memory_list[(-1)*self.memory_size:]
    
    def act(self):
        if len(self.memory_list) < self.memory_size:
            return round(random.random())
        state = int(list_to_str(np.array(self.memory_list).reshape(-1)), 2)
        qs = self.Q[state, :].reshape(-1)
        action = np.random.choice([0, 1], p=qs)
        return action
    
    def reset(self):
        self.memory_list = []


class Random_Agent(AgentBase):
    def __init__(self, memory_size=0):
        super().__init__(memory_size)
    
    def update(self, actions, reward, max_rwd=5.0):
        pass
    
    def act(self):
        return round(random.random())
    
    def reset(self):
        self.memory_list = []


class ZDGTFT_Agent(AgentBase):
    def __init__(self, memory_size=1):
        super().__init__(memory_size)
        self.Q[0,0] = 1.
        self.Q[0,1] = 0.
        self.Q[1,0] = 1./8.
        self.Q[1,1] = 7./8.
        self.Q[2,0] = 1.
        self.Q[2,1] = 0.
        self.Q[3,0] = 1./4.
        self.Q[3,1] = 3./4.
    
    def update(self, actions, reward, max_rwd=5.0):
        self.memory_list.append(actions)
        self.memory_list = self.memory_list[(-1)*self.memory_size:]
    
    def act(self):
        if len(self.memory_list) < self.memory_size:
            return round(random.random())
        state = int(list_to_str(np.array(self.memory_list).reshape(-1)), 2)
        qs = self.Q[state, :].reshape(-1)
        action = np.random.choice([0, 1], p=qs)
        return action
    
    def reset(self):
        self.memory_list = []


class QLearning_Agent(AgentBase):
    def __init__(self, memory_size=1, alpha=0.1, beta=1.0, gamma=0.95):
        super().__init__(memory_size)
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
    
    def update(self, actions, reward, max_rwd=5.0):
        if len(self.memory_list) < self.memory_size:
            self.memory_list.append(actions)
        else:
            pre_state = int(list_to_str(np.array(self.memory_list).reshape(-1)), 2)
            self.memory_list.append(actions)
            self.memory_list = self.memory_list[(-1)*self.memory_size:]
            cur_state = int(list_to_str(np.array(self.memory_list).reshape(-1)), 2)
            td_err = reward + self.gamma*np.max(self.Q[cur_state, :].reshape(-1)) - self.Q[pre_state, actions[0]]
            self.Q[pre_state, actions[0]] = self.Q[pre_state, actions[0]] + self.alpha*td_err
    
    def act(self):
        if len(self.memory_list) < self.memory_size:
            return round(random.random())
        state = int(list_to_str(np.array(self.memory_list).reshape(-1)), 2)
        qs = softmax(self.Q[state, :].reshape(-1), beta=self.beta)
        action = np.random.choice([0, 1], p=qs)
        return action
    
    def reset(self):
        self.memory_list = []
        self.Q = np.zeros((4**self.memory_size, 2))


class SR_Ctrl_Agent(AgentBase):
    def __init__(self, memory_size=1, alpha=0.1, beta=1.0, gamma=0.95, zeta=1.0):
        super().__init__(memory_size)
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.zeta = zeta
        self.n_state = 4**self.memory_size
        self.n_action = 2
        M = np.zeros((self.n_state*self.n_action, self.n_state))
        self.M = create_initial_mat(M, n_state=self.n_state, n_action=self.n_action)
    
    def update(self, actions, reward, max_rwd=5.0): # actions: [my action, his action]
        if len(self.memory_list) < self.memory_size:
            self.memory_list.append(actions)
        else:
            pre_state = int(list_to_str(np.array(self.memory_list).reshape(-1)), 2)
            self.memory_list.append(actions)
            self.memory_list = self.memory_list[(-1)*self.memory_size:]
            cur_state = int(list_to_str(np.array(self.memory_list).reshape(-1)), 2)

            next_action = np.argmax(self.Q[cur_state, :])
            pre_act_idx = pre_state*self.n_action + actions[0]
            cur_act_idx = cur_state*self.n_action + next_action
            s_rwd = reward / max_rwd
            td_err = np.exp(self.zeta*(s_rwd-1.0))*np.eye(self.n_state)[pre_state] + self.gamma*self.M[cur_act_idx, :] - self.M[pre_act_idx, :]
            self.M[pre_act_idx, :] = self.M[pre_act_idx, :] + self.alpha*td_err
            self.Q = create_q_values(self.M, n_state=self.n_state, n_action=self.n_action)
    
    def act(self):
        if len(self.memory_list) < self.memory_size:
            return round(random.random())
        state = int(list_to_str(np.array(self.memory_list).reshape(-1)), 2)
        qs = softmax(self.Q[state, :].reshape(-1), beta=self.beta)
        action = np.random.choice([0, 1], p=qs)
        return action
    
    def reset(self):
        self.memory_list = []
        self.Q = np.zeros((4**self.memory_size, 2))
        M = np.zeros((self.n_state*self.n_action, self.n_state))
        self.M = create_initial_mat(M, n_state=self.n_state, n_action=self.n_action)


class SR_Ctrl2_Agent(AgentBase):
    def __init__(self, memory_size=1, alpha=0.1, beta=1.0, gamma=0.95, zeta=1.0):
        super().__init__(memory_size)
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.zeta = zeta
        self.n_state = 4**self.memory_size
        self.n_action = 2
        M = np.zeros((self.n_state*self.n_action, self.n_state))
        self.M = create_initial_mat(M, n_state=self.n_state, n_action=self.n_action)
        self.w = np.zeros(self.n_state)
    
    def update(self, actions, reward, max_rwd=5.0): # actions: [my action, his action]
        if len(self.memory_list) < self.memory_size:
            self.memory_list.append(actions)
        else:
            pre_state = int(list_to_str(np.array(self.memory_list).reshape(-1)), 2)
            self.memory_list.append(actions)
            self.memory_list = self.memory_list[(-1)*self.memory_size:]
            cur_state = int(list_to_str(np.array(self.memory_list).reshape(-1)), 2)
            
            next_action = np.argmax(self.Q[cur_state, :])
            pre_act_idx = pre_state*self.n_action + actions[0]
            cur_act_idx = cur_state*self.n_action + next_action
            s_rwd = reward / max_rwd
            td_err = np.eye(self.n_state)[pre_state] + self.gamma*self.M[cur_act_idx, :] - self.M[pre_act_idx, :]
            self.M[pre_act_idx, :] = self.M[pre_act_idx, :] + self.alpha*td_err
            self.w[cur_state] = self.w[cur_state] + self.alpha * (s_rwd - self.w[cur_state])
            self.Q = create_q_values_w_ext_rwd(self.M, self.w, n_state=self.n_state, n_action=self.n_action, zeta=self.zeta)
    
    def act(self):
        if len(self.memory_list) < self.memory_size:
            return round(random.random())
        state = int(list_to_str(np.array(self.memory_list).reshape(-1)), 2)
        qs = softmax(self.Q[state, :].reshape(-1), beta=self.beta)
        action = np.random.choice([0, 1], p=qs)
        return action
    
    def reset(self):
        self.memory_list = []
        self.Q = np.zeros((4**self.memory_size, 2))
        M = np.zeros((self.n_state*self.n_action, self.n_state))
        self.M = create_initial_mat(M, n_state=self.n_state, n_action=self.n_action)
        self.w = np.zeros(self.n_state)


'''
Referenced from https://github.com/Axelrod-Python/Axelrod/blob/80190dbda07daf6a0ea3e5ebff1c59ab9305c24c/docs/reference/overview_of_strategies.rst#stewart-and-plotkins-tournament-2012
'''
AGENTS_DICT = {
    "all_c": All_C_Agent(),
    "all_d": All_D_Agent(),
    "extort_2": Extort_2_Agent(),
    "hard_majo": Hard_Majo_Agent(),
    "hard_joss": Hard_Joss_Agent(),
    "hard_tft": Hard_TFT_Agent(),
    "hard_tf2t": Hard_TF2T_Agent(),
    "tft": TFT_Agent(),
    "grim": Grim_Agent(),
    "gtft": GTFT_Agent(),
    "tf2t": TF2T_Agent(),
    "wsls": WSLS_Agent(),
    "random": Random_Agent(),
    "zdgtft_2": ZDGTFT_Agent(),
}