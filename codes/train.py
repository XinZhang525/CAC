import pickle
import numpy as np
import d3rlpy
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
import random
from d3rlpy.algos.dcac import *
from d3rlpy.algos.dcac_bc import *
from network import ACEncoderFactory
from data_processing import get_unsuccess_trajectories, get_trajectories

use_gpu = True if torch.cuda.is_available() else False
alpha = 0.8 # for seeking time penalty

traffic = pickle.load(open('./data/latest_traffic.pkl', 'rb'))
volume = pickle.load(open('./data/latest_volume_pickups.pkl', 'rb'))
train_airport = pickle.load(open('./data/train_airport.pkl', 'rb'))
train_trajectories = pickle.load(open('./data/train_traj_month_07_update_12.pkl', 'rb'))
expect_terminals = pickle.load(open('./data/pickup_eval.pkl', 'rb'))
unsuccess_trajecories = pickle.load(open('./data/train_traj_unsuccess_month_07.pkl', 'rb'))

train_observations, train_actions, train_rewards, train_terminals = get_trajectories(train_trajectories, alpha=alpha)
uns_observations, uns_actions, uns_rewards, uns_terminals = get_unsuccess_trajectories(unsuccess_trajecories, alpha=alpha)

all_observations = np.concatenate((train_observations, uns_observations), axis=0)
all_actions = np.concatenate((train_actions, uns_actions), axis=0)
all_rewards = np.concatenate((train_rewards, uns_rewards), axis=0)
all_terminals = np.concatenate((train_terminals, uns_terminals), axis=0)

# MDP Data loader
dataset = d3rlpy.dataset.MDPDataset(observations=all_observations, actions=all_actions, rewards=all_rewards, terminals=all_terminals,)

encoder_factory = ACEncoderFactory(64)
DCAC = d3rlpy.algos.dcac.DiscreteCAC(use_gpu=use_gpu, actor_encoder_factory=encoder_factory, critic_encoder_factory=encoder_factory, \
    n_critics=2, batch_size=64, critic_learning_rate=6.25e-5, actor_learning_rate=1e-4, temp_learning_rate=0, target_update_interval=8000, \
    alpha=0.1, initial_temperature=1, gamma=0.99) 
DCAC.fit(dataset, n_epochs=1, show_progress=True, logdir="train_log")
