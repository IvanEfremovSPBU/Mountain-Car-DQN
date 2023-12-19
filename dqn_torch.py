from collections import deque
import random
import gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Available device: ", device)

# Model: 2 input | 100 hidden | 100 hidden | 3 output
class DQN(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(state_dim, 100) 
        self.fc2 = nn.Linear(100, action_dim)
    
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        return self.fc2(x)


class Agent(object):
    def __init__(self, state_dim, action_dim):
        self.step = 0 
        self.update_freq = 200
        
        self.replay_size = 2000
        self.replay_queue = deque(maxlen=self.replay_size)
        
        self.model = DQN(state_dim, action_dim).to(device)
        self.target_model = DQN(state_dim, action_dim).to(device)
        
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)

    # Action choosing
    def act(self, s, epsilon=0.1):
        s = torch.FloatTensor(s).to(device)
        
        # Greedy strategy
        if np.random.uniform() < epsilon - self.step * 0.0002:
            return np.random.choice([0, 1, 2])
        with torch.no_grad():
            q_values = self.model(s)
        return torch.argmax(q_values).item()

    def save_model(self, file_path=f'.\models\MountainCar-v0-dqn.pth'):
        print('Model saved')
        torch.save(self.model.state_dict(), file_path)

    def remember(self, s, a, next_s, reward):
        if next_s[0] >= 0.4:
            reward += 1
        self.replay_queue.append((s, a, next_s, reward))

    def train(self, batch_size=128, lr=0.7, factor=0.9):
        # Updating target model
        if len(self.replay_queue) < self.replay_size:
            return
        self.step += 1
        if self.step % self.update_freq == 0:
            self.target_model.load_state_dict(self.model.state_dict())
        
        replay_batch = random.sample(self.replay_queue, batch_size)
        
        state_batch = torch.FloatTensor([replay[0] for replay in replay_batch]).to(device)
        next_state_batch = torch.FloatTensor([replay[2] for replay in replay_batch]).to(device)
        action_batch = torch.LongTensor([replay[1] for replay in replay_batch]).to(device)
        reward_batch = torch.FloatTensor([replay[3] for replay in replay_batch]).to(device)

        # Q values for current states
        Q_values = self.model(state_batch).gather(1, action_batch.unsqueeze(1)).squeeze(1)
        
        # Q values predictions for next states using target network
        with torch.no_grad():
            next_prediction_batch = self.target_model(next_state_batch)
        next_Q_values = next_prediction_batch.max(1)[0]
        expected_Q_values = reward_batch + factor * next_Q_values
        
        loss = nn.MSELoss()(Q_values, expected_Q_values)
        
        # Backpropagation
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

env = gym.make('MountainCar-v0')
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.n
max_episodes = 600
score_list = []
agent = Agent(state_dim, action_dim)

# Training loop
for i in range(max_episodes):
    s = env.reset()[0]
    score = 0
    while True:
        a = agent.act(s)
        next_s, reward, done, _, _ = env.step(a)
        agent.remember(s, a, next_s, reward)
        agent.train()
        score += reward
        s = next_s
        if done:
            score_list.append(score)
            print('episode:', i, 'score:', score, 'max:', max(score_list))
            break
    
    if (np.mean(score_list[-50:]) > -160 or i == max_episodes - 1) and i >= 50:
        print("Learning finished")
        agent.save_model()
        break
    
env.close()

import matplotlib.pyplot as plt

plt.plot(score_list)
plt.ylim(-10000, 0)
plt.savefig(f".\plots\Learning_score.png")