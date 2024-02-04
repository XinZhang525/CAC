import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import numpy as np
import d3rlpy
from d3rlpy.algos.dcac import *

class ActorEncoder(nn.Module):
    def __init__(self, feature_size=120):
        super(ActorEncoder, self).__init__()
        self.feature_size = feature_size
        self.conv1 = nn.Conv2d(5, 20, 3, padding=1) 
        self.pool = nn.MaxPool2d(2, 1)
        self.conv2 = nn.Conv2d(20, 30, 3)
        self.fc1 = nn.Linear(30, feature_size) 
        self.bn = nn.BatchNorm1d(feature_size)
        #self.fc2 = nn.Linear(feature_size, 84) 
        
    def forward(self, state):
        x = state.view(state.size(0), 5, 5, 5)
        x = self.pool(F.leaky_relu(self.conv1(x), 0.2))
        x = self.pool(F.leaky_relu(self.conv2(x), 0.2))
        x = x.view(-1, 30)
        x = F.leaky_relu(self.bn(self.fc1(x)))
        return x

    # THIS IS IMPORTANT!
    def get_feature_size(self):
        return self.feature_size

class CriticEncoder(nn.Module):
    def __init__(self, feature_size=120):
        super(CriticEncoder, self).__init__()
        self.feature_size = feature_size
        self.conv1 = nn.Conv2d(5, 20, 3, padding=1) 
        self.pool = nn.MaxPool2d(2, 1)
        self.conv2 = nn.Conv2d(20, 30, 3)
        self.fc1 = nn.Linear(30+19, feature_size) 
        self.bn = nn.BatchNorm1d(feature_size)
        #self.fc2 = nn.Linear(feature_size, 84) 
        
    def forward(self, state, action):
        x = state.view(state.size(0), 5, 5, 5)
        x = self.pool(F.leaky_relu(self.conv1(x), 0.2))
        x = self.pool(F.leaky_relu(self.conv2(x), 0.2))
        x = x.view(-1, 30)
        x = torch.cat([x, action], dim=1)
        x = F.leaky_relu(self.bn(self.fc1(x)))
        return x

    # THIS IS IMPORTANT!
    def get_feature_size(self):
        return self.feature_size

# For Discrete action space, actor and critic encoders are the same. 
# i.e., Actor: S --> prob_actions [bs, # actions] | Critic: S --> Q values [bs, # actions] 
class ACEncoderFactory(EncoderFactory):
    TYPE = "custom"

    def __init__(self, feature_size):
        self.feature_size = feature_size

    def create(self, observation_shape):
        return ActorEncoder(self.feature_size)

    # def create_with_action(self, observation_shape, action_size, discrete_action):
    #     return CriticEncoder(self.feature_size)

    def get_params(self, deep=False):
        return {"feature_size": self.feature_size}

