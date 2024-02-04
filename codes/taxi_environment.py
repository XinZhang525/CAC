from tqdm import tqdm
import pickle
import numpy as np
import d3rlpy
import torch

def decide_next_state(dirc, current_state):
    if current_state [0] == 0:
        return [0., 0., 0., 0.]
    if int(dirc/10) == 1:
        new_time = (current_state[2]+1)%289
    else:
        new_time = current_state[2]
    dirc = dirc%10
    
    if dirc == 0:
        new_step = [current_state[0], current_state[1]+1, new_time, current_state[-1]]
        return new_step
    if dirc == 1: 
        new_step = [current_state[0]+1, current_state[1]+1, new_time, current_state[-1]] 
        return new_step
    if dirc == 2:
        new_step = [current_state[0]+1, current_state[1], new_time, current_state[-1]]
        return new_step
    if dirc == 3: 
        new_step = [current_state[0]+1, current_state[1]-1, new_time, current_state[-1]]
        return new_step
    if dirc == 4:
        new_step = [current_state[0], current_state[1]-1, new_time, current_state[-1]]
        return new_step
    if dirc == 5: 
        new_step = [current_state[0]-1, current_state[1]-1, new_time, current_state[-1]]
        return new_step
    if dirc == 6:
        new_step = [current_state[0]-1, current_state[1], new_time, current_state[-1]]
        return new_step
    if dirc == 7:
        new_step = [current_state[0]-1, current_state[1]+1, new_time, current_state[-1]]
        return new_step
    if dirc == 8:
        new_step = [current_state[0], current_state[1], new_time, current_state[-1]]
        return new_step
    else:
        return [0., 0., 0., 0.]


def processing_state_features(input_state, volume, train_airport, traffic):
    '''
    input_state = (x, y, t, day)
    output = [5*5, 5*5, 5*5, 5*5, 5*5]
    '''
    x = int(input_state[0])
    y = int(input_state[1])
    t = int(input_state[2])
    day = int(input_state[3])

    if x == y == t == day == 0:
        return [0. for x in range(121)]

    x_range = list(range(x-2, x+3))
    y_range = list(range(y-2, y+3))
    
    # state features
    n_p = []
    n_v = []
    t_s = []
    t_w = []
    for i in x_range:
        for j in y_range:
            if (i, j, t, day) in volume:
                n_p.append(volume[(i, j, t, day)][0])
                n_v.append(volume[(i, j, t, day)][1])
            else:
                n_p.append(0.)
                n_v.append(0.)
                
            if (i, j, t, day) in traffic:
                t_s.append(traffic[(i, j, t, day)][0])
                t_w.append(traffic[(i, j, t, day)][1])
            else:
                t_s.append(0.)
                t_w.append(0.)
                
    ta = []
    for place in train_airport:
        ta.append(abs(x - train_airport[place][0][0]) + abs(y - train_airport[place][0][1]))
    
    whole_step = []
    whole_step.extend(ta)
    whole_step.extend(n_p)
    whole_step.extend(n_v)
    whole_step.extend(t_s)
    whole_step.extend(t_w)
    return whole_step



class TaxiEnv(): 
    def __init__(self, volume, train_airport, traffic, expect_terminals, initial_states):
        self.volume = volume
        self.train_airport =train_airport
        self.traffic = traffic
        self.expect_terminals = expect_terminals
        self.initial_states = initial_states
        self.action_space = len(range(0,19))

    def sample_action(self):
        return np.random.choice(self.action_space, 1)[0]

    def seed(self, seed=None):
        return np.random.seed(seed)
        
    def reset(self):  # just for trying purpose 
        return self.initial_states[np.random.choice(len(self.initial_states), 1)].reshape(-1,)

    def init_test_states(self):
        return self.initial_states

    def step(self, state, action):  # state.shape --> (4,), action --> int
        next_state = np.array(decide_next_state(action, state))
        terminate = False
        reward = 0

        if next_state[2] < state[2]: # if new day
            next_state[3] = (next_state[3] +1)%7

        if int(action) == 8:
            terminate = True
            if tuple(state) in self.expect_terminals:
                reward = self.expect_terminals[tuple(state)][3]

        return next_state, reward, terminate


