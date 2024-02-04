import random
import numpy as np

def reward_clipping(reward):
    if reward > 160:
        reward = 160
    return reward

def get_unsuccess_trajectories(x_trajects, alpha=0.8):
    '''extract transitions from grid_trajectories for success trajectories
       return list of states, list of actions, list of rewards, list of terminals
    '''
    stack_states = []
    stack_actions = []
    stack_terminals = []
    stack_sparse_rewards = []
    for i in range(len(x_trajects)):
        # episode_traj size (time_step, 130) --> sate_features from cols 0-124, action at col 125, and reward at col 129
        episode_traj = np.array(x_trajects[i])   
        episode_states = episode_traj[:,:-5]
        episode_states = np.insert(episode_states, -1, episode_traj[-1,:-5], axis=0)
        episode_actions = episode_traj[:,-5]
        episode_actions[-1] = 8  
        episode_actions  = np.insert(episode_actions, -1, 8, axis=0)
        episode_terminal = np.zeros(len(episode_traj)+1, dtype=np.int16)  
        episode_terminal[len(episode_traj)] = 1 

        reward = -4
        episode_rewards = np.where(episode_actions >= 10, -alpha, 0) # -alpha for actions that takes time.
        episode_rewards[-1] = reward  
        episode_rewards[-2] = reward  
        
        # stacking trajectories to 1D-vector corresponding to states, actions, rewards, and terminals
        stack_states.extend(episode_states)
        stack_actions.extend(episode_actions)
        stack_terminals.extend(episode_terminal)
        stack_sparse_rewards.extend(episode_rewards)
    return np.array(stack_states), np.array(stack_actions, dtype=np.int16), np.array(stack_sparse_rewards)*2, np.array(stack_terminals)

def get_trajectories(x_trajects, alpha=0.8, beta=2):
    '''extract transitions from grid_trajectories for successful trajectories
       return list of states, list of actions, list of rewards, list of terminals
    '''
    stack_states = []
    stack_actions = []
    stack_terminals = []
    stack_sparse_rewards = []
    
    for i in range(len(x_trajects)):
        # episode_traj size (time_step, 130) --> sate_features from cols 0-124, action at col 125, and reward at col 129
        episode_traj = np.array(x_trajects[i]) 
        episode_states = episode_traj[:,:-5] 
        episode_states  = np.insert(episode_states, -1, episode_traj[-1,:-5], axis=0) 
        episode_actions = episode_traj[:,-5]
        episode_actions  = np.insert(episode_actions, -1, episode_traj[-1,-5], axis=0)
        episode_actions  = np.insert(episode_actions, -1, episode_traj[-1,-5], axis=0)
        episode_terminal = np.zeros(len(episode_traj)+1, dtype=np.int16)  
        episode_terminal[len(episode_traj)] = 1  # set the termination of an episode to 1

        reward = episode_traj[-1,-1]
        cliped_reward = (beta-alpha)*reward_clipping(reward)
        episode_rewards  = np.where(episode_actions >= 10, -alpha, 0) # -alpha for actions that takes time.
        episode_rewards[-1] = cliped_reward
        episode_rewards[-2] = cliped_reward


        # stacking trajectories to 1D-vector corresponding to states, actions, rewards, and terminals
        stack_states.extend(episode_states)
        stack_actions.extend(episode_actions)
        stack_terminals.extend(episode_terminal)
        stack_sparse_rewards.extend(episode_rewards)
        
    return np.array(stack_states), np.array(stack_actions, dtype=np.int16), np.array(stack_sparse_rewards)*2, np.array(stack_terminals)



