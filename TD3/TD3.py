import numpy as np
import torch
import random
from collections import deque
import copy
import gym
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
import sys
from pyvirtualdisplay import Display

# Config
ENV = "BipedalWalkerHardcore-v3"
BATCH_SIZE = 100
DISCOUNT_FACTOR = 0.99
EXPLORE_POLICY = 0.1
LEARN_RATE = .001
POLICY_DELAY = 2
TAU = 0.005
NOISE_POLICY = 0.2
NOISE_CLIP = 0.5

class ExperienceReplay:


    def __init__(self, buffer_size, batch_size, device):
        self.buffer = deque(maxlen=buffer_size)
        self.batch_size= batch_size
        self.device = device
        self.ptr = 0
        print(self.buffer.maxlen)

    def __len__(self):
        return len(self.buffer)

    # Add a transition to the memory by basic SARNS convention.
    def store_transition(self, state, action, reward, new_state, done):
        # If buffer is abuot to overflow, begin rewriting existing memory?
        if self.ptr < self.buffer.maxlen:
            self.buffer.append((state, action, reward, new_state, done))
        else:
            self.buffer[int(self.ptr)] = (state, action, reward, new_state, done)
            self.ptr = (self.ptr + 1) % self.buffer.maxlen


    # Sample only the memory that has been stored. Samples BATCH
    # amount of samples.
    def sample(self):
        sample = random.sample(self.buffer, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*sample)
        states = torch.from_numpy(np.array(states, dtype=np.float32)).to(self.device)
        actions = torch.from_numpy(np.array(actions, dtype=np.float32)).to(self.device)
        rewards = torch.from_numpy(np.array(rewards, dtype=np.float32).reshape(-1, 1)).to(self.device)
        next_states = torch.from_numpy(np.array(next_states, dtype=np.float32)).to(self.device)
        dones = torch.from_numpy(np.array(dones, dtype=np.uint8).reshape(-1, 1)).float().to(self.device)
        return (states, actions, rewards, next_states, dones)
    
class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, max_actions):
        super(Actor, self).__init__()

        # Make a simple 3 later linear network
        self.l1 = nn.Linear(state_dim, 400)
        self.l2 = nn.Linear(400, 300)
        self.l3 = nn.Linear(300, action_dim)
        self.max_actions = max_actions

    def forward(self, state):
        x = F.relu(self.l1(state))
        x = F.relu(self.l2(x))
        x = self.max_actions * torch.tanh(self.l3(x))
        return x


class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        # Defined the Q1 and Q2 of the TD3.
        # https://arxiv.org/pdf/1802.09477.pdf
        super(Critic, self).__init__()
        # Q1. Final layer of Q1 to return single value.
        self.l1 = nn.Linear(state_dim + action_dim, 400)
        self.l2 = nn.Linear(400, 300)
        self.l3 = nn.Linear(300, 1)

        # Q2. Same as Q1.
        self.l4 = nn.Linear(state_dim + action_dim, 400)
        self.l5 = nn.Linear(400, 300)
        self.l6 = nn.Linear(300, 1)

    def forward(self, state, action):
        # Perform forward pass through NN with the given state
        # and the action to take on this state.
        # Flatten state + action so we have the input value shape to
        # pass to the NN.
        sa = torch.cat([state, action], 1)

        # Q1 value computation.
        c1 = F.relu(self.l1(sa))
        c1 = F.relu(self.l2(c1))
        c1 = self.l3(c1)

        # Q2 value computation.
        c2 = F.relu(self.l4(sa))
        c2 = F.relu(self.l5(c2))
        c2 = self.l6(c2)

        # Return both values so we can grab the min of the two.
        return (c1, c2)

class TD3():
    def __init__(self, state_dim, action_dim, max_action, env, device):
        super(TD3, self).__init__()

        # Set up Actor net
        self.actor = Actor(state_dim, action_dim, max_action).to(device)
        self.actor_target = copy.deepcopy(self.actor)
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=LEARN_RATE)
        self.device = device

        # Set up Critic net
        self.critic = Critic(state_dim, action_dim).to(device) # only needs state + action
        self.critic_target = copy.deepcopy(self.critic)
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=LEARN_RATE)
        self.max_action = max_action
        self.env = env

    def select_action(self, state, noise=0.1):
        # Gets best action to take based on current state/policy.
        state = torch.FloatTensor(state.reshape(1, -1)).to(self.device)
        action = self.actor(state).cpu().data.numpy().flatten()
        # Add a random amount of noise from a normal dist to the action.
        if(noise == EXPLORE_POLICY):
            action = (action + np.random.normal(0, noise, size=self.env.action_space.shape[0]))

        return self.actor(state).cpu().data.numpy().flatten()



    def train(self, replay_buffer, current_iteration):
        # Pseudocode detailed by :
        # http://bicmr.pku.edu.cn/~wenzw/bigdata/lect-dyna3w.pdf

        # Randomly sample batch (n = 100) of transitions from replay replay_buffer D.
        # All SARNS + done are already Tensors.
        state, action, reward, next_state, done = replay_buffer.sample()
        # Find the target action.
        # noise = sampled from N(0, sigma), where sigma = NOISE_POLICY (0.2).
        # Clips all values to be in the range of noise_clip (-0.5, 0.5).
        # https://stackoverflow.com/questions/44417227/make-values-in-tensor-fit-in-given-range
        tensor_cpy = action.clone().detach()
        noise = tensor_cpy.normal_(0, NOISE_POLICY).clamp(-NOISE_CLIP, NOISE_CLIP)
        # noise = (torch.randn_like(action) * NOISE_POLICY).clamp(-NOISE_CLIP, NOISE_CLIP)

        # Clips the next action + clipped_noise with -max_action, +max_action.
        next_action = (self.actor_target(next_state) + noise).clamp(-self.max_action, self.max_action)
        # Compute target Qs:
        # Runs forward pass with the next_state and the next_action, returns (Q1, Q2).
        # Softmax? Of Q1, Q2 (min i=1,2 of Q_target_i(s',a'(s')))
        target_q1, target_q2 = self.critic_target(next_state, next_action)
        target_q = ((torch.min(target_q1, target_q2)) * (1-done)) + reward
        curr_q1, curr_q2 = self.critic(state, action)

        # ... and then both are learned by regressing (MSE) to the target:
        critic_loss = F.mse_loss(curr_q1, target_q) + F.mse_loss(curr_q2, target_q)
        self.critic_optimizer.zero_grad() # reset any previously held grads to 0, else it accumulates
        critic_loss.backward()
        self.critic_optimizer.step() # Updates Q-functions by one gradient step.

        # The policy is learned by maximizing the Q every other iteration
        if (current_iteration % POLICY_DELAY == 0):
            # Update policy by one step of grad ascent.
            # 1/|batch_size| sum(-self.critic(state, self.actor(state))[0])
            # self.critic(...)[0] gets Q1 calculation.
            actor_loss = -self.critic(state, self.actor(state))[0].mean()

            # Update target networks:
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            # If i % policy_delay == 0, then we update model (delayed updates)
            for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
                target_param.data.copy_(TAU * param.data + (1 - TAU) * target_param.data)

            for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
                target_param.data.copy_(TAU * param.data + (1 - TAU) * target_param.data)
