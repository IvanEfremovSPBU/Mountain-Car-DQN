import gym
import torch
from torch import nn
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as st
from data_plotting import *


class DQN(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(state_dim, 100)
        self.fc2 = nn.Linear(100, action_dim)
    
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        return self.fc2(x)


env = gym.make('MountainCar-v0')
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.n

# Init model
model = DQN(state_dim, action_dim)

# Load model
model.load_state_dict(torch.load(f'.\models\MountainCar-v0-dqn.pth'))
model.eval()


# Function to distinguish between plotting startup and movement emulation
# run => show movement emulation
# plot => construct plots of num_episodes emulations
def start_env(num_episodes=1, mode='run'):
    
    modes = ['run', 'plot']
    if mode not in modes:
        raise ValueError("Invalid mode type. Expected one of: %s" % modes)
    
    # If emulation show the car
    if mode == 'run':
        env = gym.make('MountainCar-v0', render_mode='human')
    else:
        env = gym.make('MountainCar-v0')
    
    rewards = []
    means = []
    
    for episode in range(num_episodes):
        s = env.reset()[0]
        total_reward = 0
        done = False
        
        while not done:
            env.render()
            
            state_tensor = torch.FloatTensor(s)
            
            with torch.no_grad():
                action = model(state_tensor).argmax().item()
            
            next_state, reward, done, _, _ = env.step(action)
            
            total_reward += reward
            s = next_state
        
        env.render()
        
        # Collect plotting data
        if mode == 'plot':
            if episode % 1000 == 0:
                print("Episode:", episode, "Total Reward:", total_reward)
            
            rewards.append(total_reward)
            
            if episode % 10 == 0 and episode > 0:
                means.append(np.mean(rewards[episode-10:episode]))

    env.close()
    
    if mode == 'plot':
        plot_score(rewards, means)
        plot_range(rewards, means)
        plot_percentiles(rewards)
        plot_conf_interval(rewards)


start_env()


