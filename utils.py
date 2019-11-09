import numpy as np
import os, sys


def list_to_str(in_list):
    out = ""
    for tmp_v in in_list:
        out = out + str(tmp_v)
    return out


def softmax(a, beta=1.0):
    c = np.max(a)
    exp_a = np.exp(a*beta)
    sum_exp_a = np.sum(exp_a)
    y = exp_a / sum_exp_a
    return y


'''
Creating a matrix \overline{M}(s, a', s')
'''
def create_q_func_mat(_M, n_state=16, n_action=4):
    tmp_M = np.copy(_M)
    for i in range(n_state):
        for j in range(n_action):
            tmp_M[i*n_action+j, i] = 0.0
    return tmp_M

def create_initial_mat(_M, n_state=16, n_action=4):
    tmp_M = np.copy(_M)
    for i in range(n_state):
        for j in range(n_action):
            tmp_M[i*n_action+j, i] = 1.0
    return tmp_M


def compute_intrisic_rewards(q_M, n_state=16, n_action=4):
    out_vec = np.zeros(n_state)
    for i in range(n_state):
        tmp_v = 0.0
        for j in range(n_action):
            tmp_v = tmp_v + np.sum(q_M[n_action*i + j])
        out_vec[i] = tmp_v
    return out_vec


def create_q_values(M, n_state=16, n_action=4):
    q_M = create_q_func_mat(M, n_state=n_state, n_action=n_action)
    int_rwds = compute_intrisic_rewards(q_M, n_state=n_state, n_action=n_action)
    q_vals = []
    for i in range(n_state*n_action):
        q_vals.append(np.dot(q_M[i,:], int_rwds))
    q_vals = np.array(q_vals).reshape(n_state, n_action)
    return q_vals


def create_q_values_w_ext_rwd(M, w, n_state=16, n_action=4, zeta=1.0):
    q_M = create_q_func_mat(M, n_state=n_state, n_action=n_action)
    int_rwds = compute_intrisic_rewards(q_M, n_state=n_state, n_action=n_action)
    rwds = int_rwds + zeta*w
    q_vals = []
    for i in range(n_state*n_action):
        q_vals.append(np.dot(q_M[i,:], rwds))
    q_vals = np.array(q_vals).reshape(n_state, n_action)
    return q_vals


def compute_mean_action_sr(M, obs_id, n_state=4**5, n_action=2):
    out_vec = np.zeros(n_state)
    for i in range(n_action):
        out_vec = out_vec + M[obs_id*n_action+i, :]
    out_vec = out_vec / float(n_action)
    return out_vec


'''
me\op 0(C) 1(D)
0(C)    3,   0
1(D)    5,   1
'''
RWD_MAP = np.array([
    [3, 0],
    [5, 1]
])