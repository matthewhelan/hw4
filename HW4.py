import gym
import mujoco_py

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import random
import numpy as np
import matplotlib.pyplot as plt
import copy

env = gym.make('Hopper-v3')

Na = env.action_space.shape[0]
A_MAX = env.action_space.high[0]
A_MIN = env.action_space.low[0]
Ns = env.observation_space.shape[0]
EPISODE = 5000
BUFFER_SIZE = 1e5
BATCH_SIZE = 256
GAMMA = 0.99
LR_C = 1e-3
LR_A = 1e-4
TAU = 1e-3
SIGMA = 0.02
POLICY_UPDATE_FREQ = 5


def plot_durations(episode_index, episode_return):
    plt.figure(1)
    plt.clf()
    plt.title('Training...')
    plt.xlabel('Episode')
    plt.ylabel('Return')
    plt.plot(episode_index, episode_return)
    plt.pause(0.001)  # pause a bit so that plots are updated

class ReplayBuffer:
    def __init__(self, BUFFER_SIZE):
        self.buffer_size = BUFFER_SIZE
        self.memory = []

    def push(self, data):
        self.memory.append(data)
        if len(self.memory) > self.buffer_size:
            del self.memory[0]

    def sample(self, BATCH_SIZE):
        return random.sample(self.memory, BATCH_SIZE)

    def __len__(self):
        return len(self.memory)

class ActorNet(nn.Module):
    def __init__(self):
        super(ActorNet, self).__init__()
        self.fc1 = nn.Linear(Ns, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, Na)
        self.tanh = nn.Tanh()

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.tanh(self.fc3(x))
        return x

class CriticNet(nn.Module):
    def __init__(self):
        super(CriticNet, self).__init__()
        self.fc1 = nn.Linear(Ns + Na, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 1)
        self.fc4 = nn.Linear(Ns + Na, 256)
        self.fc5 = nn.Linear(256, 256)
        self.fc6 = nn.Linear(256, 1)

    def forward(self, xs):
        x, a = xs
        x = F.relu(self.fc1(torch.cat([x, a], 1)))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        x = x.reshape([-1])
        return x

actor_target_net = ActorNet()
actor_net = ActorNet()  # copy.deepcopy(actor_target_net)
critic_target_net = CriticNet()
critic_net = CriticNet()  # copy.deepcopy(critic_target_net)

# Other two critic networks only for TD3
critic_net2 = CriticNet()
critic_target_net2 = CriticNet()

replay_buffer = ReplayBuffer(BUFFER_SIZE)

optimizer_critic = optim.Adam(critic_net.parameters(), lr=LR_C)  
optimizer_critic2 = optim.Adam(critic_net2.parameters(), lr=LR_C)  #  second optimizer for TD3 only
optimizer_actor = optim.Adam(actor_net.parameters(), lr=LR_A)

def select_action(obs, noiseless=False):
    # TODO: pick action according to actor_net and Gaussian noise
    # If noiseless=True, do not add noise (for evaluation purpose) 
    # obs = torch.from_numpy(obs).float().unsqueeze(0)
    # Get action prediction
    action = actor_net(obs).detach()

    # Add Gaussian noise to the action if not in noiseless mode
    if not noiseless:
        noise = torch.normal(mean=torch.zeros(Na), std=SIGMA)
        action += noise

    # Clamp the action to the valid action space limits and convert to numpy for the environment
    action = torch.clamp(action, A_MIN, A_MAX)
    return torch.tensor(action, dtype=torch.float)  
    # action = np.random.normal(SIGMA, size=(Na,))
    # return torch.tensor(action, dtype=torch.float)

def train_DDPM():
    if len(replay_buffer) < BATCH_SIZE:
        return
    else:
        sample_batch = replay_buffer.sample(BATCH_SIZE)
    s, a, r, _s, D = zip(*sample_batch)
    state_batch = torch.stack(s)
    action_batch = torch.stack(a)
    reward_batch = torch.tensor(r, dtype=torch.float32)
    _state_batch = torch.stack(_s)
    done_batch = torch.tensor(D, dtype=torch.float32)

    # TODO: calculate critic_loss and perform gradient update
    optimizer_critic.zero_grad()
    # Predict the current Q-values
    current_Q_values = critic_net((state_batch, action_batch))
    # Compute the next actions using the actor target network
    next_actions = actor_target_net(_state_batch)
    # Predict the next Q-values using the critic target network
    next_Q_values = critic_target_net((_state_batch, next_actions))
    # Compute the target Q-values
    target_Q_values = reward_batch + GAMMA * next_Q_values * (1 - done_batch)
    # Compute the critic loss
    critic_loss = F.mse_loss(current_Q_values, target_Q_values)
    # Backpropagate the loss
    critic_loss.backward()
    optimizer_critic.step()

    # TODO: calculate actor_loss and perform gradient update
    optimizer_actor.zero_grad()
    # Compute the actor loss
    predicted_actions = actor_net(state_batch)
    # Gradient ASCENT so negative loss, average loss across samples
    actor_loss = -critic_net((state_batch, predicted_actions)).mean()
    # Backpropagate the loss
    actor_loss.backward()
    optimizer_actor.step()

    # TODO: update target networks
    # Same as HW3 target update
    for target_param, param in zip(actor_target_net.parameters(), actor_net.parameters()):
        target_param.data.copy_(target_param.data * (1.0 - TAU) + param.data * TAU)

    for target_param, param in zip(critic_target_net.parameters(), critic_net.parameters()):
        target_param.data.copy_(target_param.data * (1.0 - TAU) + param.data * TAU)

def train_TD3():
    if len(replay_buffer) < BATCH_SIZE:
        return
    
    sample_batch = replay_buffer.sample(BATCH_SIZE)
    s, a, r, _s, D = zip(*sample_batch)
    state_batch = torch.stack(s)
    action_batch = torch.stack(a)
    reward_batch = torch.tensor(r, dtype=torch.float32)
    _state_batch = torch.stack(_s)
    done_batch = torch.tensor(D, dtype=torch.float32)

    with torch.no_grad():
        next_actions = actor_target_net(_state_batch)
        noise = (torch.randn_like(action_batch) * SIGMA).clamp(-0.5, 0.5)
        next_actions = (next_actions + noise).clamp(A_MIN, A_MAX)
        
        target_Q1 = critic_target_net((_state_batch, next_actions))
        target_Q2 = critic_target_net2((_state_batch, next_actions))
        target_Q = reward_batch + GAMMA * (1 - done_batch) * torch.min(target_Q1, target_Q2)

    current_Q1 = critic_net((state_batch, action_batch))
    current_Q2 = critic_net2((state_batch, action_batch))
    critic1_loss = F.mse_loss(current_Q1, target_Q)
    critic2_loss = F.mse_loss(current_Q2, target_Q)

    optimizer_critic.zero_grad()
    critic1_loss.backward()
    optimizer_critic.step()

    optimizer_critic2.zero_grad()
    critic2_loss.backward()
    optimizer_critic2.step()

    # Update only if t mod n = 0
    if episode % POLICY_UPDATE_FREQ == 0:
        actor_loss = -critic_net((state_batch, actor_net(state_batch))).mean()
        optimizer_actor.zero_grad()
        actor_loss.backward()
        optimizer_actor.step()

        # Update target networks like DDPM
        for target_param, param in zip(actor_target_net.parameters(), actor_net.parameters()):
            target_param.data.copy_(target_param.data * (1.0 - TAU) + param.data * TAU)

        for target_param, param in zip(critic_target_net.parameters(), critic_net.parameters()):
            target_param.data.copy_(target_param.data * (1.0 - TAU) + param.data * TAU)

        for target_param, param in zip(critic_target_net2.parameters(), critic_net2.parameters()):
            target_param.data.copy_(target_param.data * (1.0 - TAU) + param.data * TAU)
       
       

timer = 0
R = 0
Return = []
episode_indexes = []

for episode in range(EPISODE):
    obs = env.reset()
    obs = torch.tensor(obs, dtype=torch.float)
    done = False
    timer = 0
    R = 0
    eval = (episode % 10 == 0)

    while not done:
        action = select_action(obs, eval)
        action = torch.clamp(action, min=A_MIN, max=A_MAX)
        obs_, reward, terminated, _ = env.step(action.numpy())
        done = terminated
        obs_ = torch.tensor(obs_, dtype=torch.float)
        transition = (obs, action, reward, obs_, done)
        replay_buffer.push(transition)
        # train_DDPM()
        train_TD3()
        R += reward
        timer += 1
        obs = obs_

    if eval:  # evaluation without noise
        print('Episode: %3d,\tStep: %5d,\tReturn: %f' %(episode, timer, R))
        Return.append(R)
        episode_indexes.append(episode)
        plot_durations(episode_indexes, Return)

plt.show()
