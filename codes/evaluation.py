import pickle
import numpy as np
import d3rlpy
from tqdm import tqdm
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import random
from d3rlpy.algos.dcac import *
from d3rlpy.algos.dcac_bc import *
from network import ACEncoderFactory
from data_processing import get_unsuccess_trajectories, get_trajectories
from taxi_environment import TaxiEnv, processing_state_features


def evaluate_CAC_on_test_inits_until_found(model, env, max_len, render=False, random_action=False):
    rewards = []
    seek_times = []
    successes = []
    states = env.init_test_states()
    for state in tqdm(states):
        Thr = 0 # not terminate in the first step.
        if render:
            print(f"inital state: {state}")
        seek_time = 0
        j = 0
        while j <= max_len:
            policy_input = state.tolist()
            policy_input.extend(processing_state_features(state, volume, train_airport, traffic))
            policy_input = np.array(policy_input)  # state features shape (125,)

            if random_action is True:
                action = env.sample_action()
                if action == 9:
                    action = env.sample_action()
            else:
                torch_inp = torch.unsqueeze(torch.from_numpy(policy_input), 0)
                _, log_prob = model.predict_with_prob(torch_inp.to(torch.float32))
                probabilities = np.exp(log_prob[0].cpu().detach().numpy())
                if Thr == 0:
                    probabilities[8] = 0
                    probabilities /= probabilities.sum()
                action = np.random.choice(19, 1, p=probabilities)[0]

            Thr += 1
            next_state, reward, terminate = env.step(state, action)
            if np.array_equal(next_state, np.array([0., 0., 0., 0.])):
                seek_time = max_len
                # print(f'{k} state: 0,0,0,0')
                break
            if next_state[3] != state[3]:  # abs(next_state[2] - state[2]) > 287:
                seek_time += 1
            else:
                seek_time += abs(next_state[2] - state[2])
            state = next_state

            if render:
                print(f" Action at step {j}: {action} --> {state}, reward={round(reward, 2)}, terminal={terminate}")

            if terminate is True and reward != 0:
                break

            if terminate is True and reward == 0:  # waiting in that state for 5 minutes (action = 18)
                Thr = 0
                next_state, _, _ = env.step(state, 18)
                seek_time += 1
                state = next_state
                if render:
                    print(f" Action at step {j}: {action} --> {state}, reward={round(reward, 2)}, terminal={terminate}")
            j += 1
        seek_time += 1
        rewards.append(reward)
        seek_times.append(seek_time)
        successes.append(1 if reward > 0 else 0)
    return rewards, seek_times, successes


def print_eval_metrics(rewards, seek_times, successes):
    print('Average money recieved: %.3f' %np.mean(rewards))
    print('Average seeking time: %.3f' %np.mean(seek_times))
    print('Earning Efficiency: %.3f' %(np.sum(rewards)/np.sum(seek_times)))
    print('Success Rate: %.3f' %np.mean(successes))

use_gpu = True if torch.cuda.is_available() else False
traffic = pickle.load(open('./data/latest_traffic.pkl', 'rb'))
volume = pickle.load(open('./data/latest_volume_pickups.pkl', 'rb'))
train_airport = pickle.load(open('./data/train_airport.pkl', 'rb'))
test_trajectories = pickle.load(open('./data/test_traj_month_07.pkl', 'rb'))
expect_terminals = pickle.load(open('./data/pickup_eval.pkl', 'rb'))

test_observations, test_actions, test_rewards, test_terminals = get_trajectories(test_trajectories, alpha=0.8)
test_dataset = d3rlpy.dataset.MDPDataset(observations=test_observations, actions=test_actions, rewards=test_rewards, terminals=test_terminals)

# set all possible initial points to all initil points of trajectories in test set
test_terminal_indices = np.where(test_terminals == 1)[0]
test_start_indices = test_terminal_indices[:-1] + 1
test_start_indices = np.append(test_start_indices, 0)

env = TaxiEnv(volume, train_airport, traffic, expect_terminals=expect_terminals, initial_states= test_observations[test_start_indices][:, :4])

# load CAC model for evaluation
encoder_factory = ACEncoderFactory(64)
DCAC = d3rlpy.algos.dcac.DiscreteCAC(use_gpu=use_gpu, actor_encoder_factory=encoder_factory, critic_encoder_factory=encoder_factory, \
    n_critics=2, batch_size=64, critic_learning_rate=6.25e-5, actor_learning_rate=1e-4, temp_learning_rate=0, target_update_interval=8000, \
    alpha=0.1) 
DCAC.build_with_dataset(test_dataset)
DCAC.load_model('train_log/DiscreteCAC_20230117172321/model_88127.pt') 

dcac_rewards, dcac_seek_times, dcac_successes = evaluate_CAC_on_test_inits_until_found(model=DCAC, env=env, max_len=60, render=False)
print_eval_metrics(dcac_rewards, dcac_seek_times, dcac_successes)

