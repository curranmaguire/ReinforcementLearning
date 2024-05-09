# https://github.com/ikostrikov/pytorch-a2c-ppo-acktr-gail/blob/master/a2c_ppo_acktr/algo/a2c_acktr.py
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import gym
from collections import namedtuple, deque
import torch.optim as optim
import random
import matplotlib.pyplot as plt
import os
import time


class Actor(nn.Module):
    def __init__(
        self, state_size=24, action_size=4, seed=42, hidden_size=256, num_layers=2
    ):
        super(Actor, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.lstm = nn.LSTM(state_size, hidden_size, num_layers, batch_first=True)
        self.fc1 = nn.Linear(hidden_size, 400)
        self.fc2 = nn.Linear(400, 300)
        self.fc3 = nn.Linear(300, action_size)

    def forward(self, state):
        """Build an actor (policy) network that maps states -> actions."""

        out, hidden = self.lstm(state, self.init_hidden(state.shape[0]))
        x = F.relu(self.fc1(out[:, -1, :]))
        x = F.relu(self.fc2(x))
        return F.torch.tanh(self.fc3(x))

    def init_hidden(self, batch_size):
        return (
            torch.zeros(self.num_layers, batch_size, self.hidden_size),
            torch.zeros(self.num_layers, batch_size, self.hidden_size),
        )


# Q1-Q2-Critic Neural Network


class Critic(nn.Module):
    def __init__(self, state_size, action_size, seed=42, hidden_size=256, num_layers=2):
        super(Critic, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.lstm = nn.LSTM(
            state_size + action_size, hidden_size, num_layers, batch_first=True
        )
        self.fc1 = nn.Linear(hidden_size, 400)
        self.fc2 = nn.Linear(400, 300)
        self.fc3 = nn.Linear(300, 1)

    def forward(self, state, action):
        """Build a critic (value) network that maps (state, action) pairs -> Q-values."""

        xa = torch.cat([state, action], dim=-1)
        out, hidden = self.lstm(xa, self.init_hidden(xa.shape[0]))
        x = F.relu(self.fc1(out[:, -1, :]))
        x = F.relu(self.fc2(x))
        return self.fc3(x)

    def init_hidden(self, batch_size):
        return (
            torch.zeros(self.num_layers, batch_size, self.hidden_size),
            torch.zeros(self.num_layers, batch_size, self.hidden_size),
        )


class SysModel(nn.Module):
    def __init__(self, state_size, action_size, seed=42, hidden_size=256, num_layers=2):
        super(SysModel, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.lstm = nn.LSTM(
            state_size + action_size, hidden_size, num_layers, batch_first=True
        )
        self.fc1 = nn.Linear(hidden_size, 400)
        self.fc2 = nn.Linear(400, 300)
        self.fc3 = nn.Linear(300, state_size)

    def forward(self, state, action):
        """Build a critic (value) network that maps (state, action) pairs -> Q-values."""
        xa = torch.cat([state, action], dim=-1)
        out, hidden = self.lstm(xa, self.init_hidden(xa.shape[0]))
        x = F.relu(self.fc1(out[:, -1, :]))
        x = F.relu(self.fc2(x))
        return self.fc3(x)

    def init_hidden(self, batch_size):
        return (
            torch.zeros(self.num_layers, batch_size, self.hidden_size),
            torch.zeros(self.num_layers, batch_size, self.hidden_size),
        )


class TD3_FORK:
    def __init__(
        self,
        name,
        env,
        load=False,
        gamma=0.99,  # discount factor
        lr_actor=3e-4,
        lr_critic=3e-4,
        lr_sysmodel=3e-4,
        batch_size=100,
        buffer_capacity=1000000,
        tau=0.02,  # target network update factor
        random_seed=np.random.randint(1, 10000),
        cuda=True,
        policy_noise=0.2,
        std_noise=0.1,
        noise_clip=0.5,
        policy_freq=2,  # target network update period
    ):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.env = env
        self.create_actor()
        self.create_critic()
        self.create_sysmodel()
        self.act_opt = optim.Adam(self.actor.parameters(), lr=lr_actor)
        self.crt_opt = optim.Adam(self.critic.parameters(), lr=lr_critic)
        self.sys_opt = optim.Adam(self.sysmodel.parameters(), lr=lr_sysmodel)
        self.set_weights()
        self.replay_memory_buffer = deque(maxlen=buffer_capacity)
        self.replay_memory_bufferd_dis = deque(maxlen=buffer_capacity)
        self.batch_size = batch_size
        self.tau = tau
        self.policy_freq = policy_freq
        self.gamma = gamma
        self.name = name
        self.upper_bound = self.env.action_space.high[0]  # action space upper bound
        self.lower_bound = self.env.action_space.low[0]  # action space lower bound
        self.obs_upper_bound = self.env.observation_space.high[
            0
        ]  # state space upper bound
        self.obs_lower_bound = self.env.observation_space.low[
            0
        ]  # state space lower bound
        self.policy_noise = policy_noise
        self.noise_clip = noise_clip
        self.std_noise = std_noise

    def create_actor(self):
        params = {
            "state_size": self.env.observation_space.shape[0],
            "action_size": self.env.action_space.shape[0],
            "seed": 42,
        }
        self.actor = Actor(**params).to(self.device)
        self.actor_target = Actor(**params).to(self.device)

    def create_critic(self):
        params = {
            "state_size": self.env.observation_space.shape[0],
            "action_size": self.env.action_space.shape[0],
            "seed": 42,
        }
        self.critic = Critic(**params).to(self.device)
        self.critic_target = Critic(**params).to(self.device)

    def create_sysmodel(self):
        params = {
            "state_size": self.env.observation_space.shape[0],
            "action_size": self.env.action_space.shape[0],
        }
        self.sysmodel = SysModel(**params).to(self.device)

    def set_weights(self):
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.critic_target.load_state_dict(self.critic.state_dict())

    def load_weight(self):
        self.actor.load_state_dict(
            torch.load(
                "/content/drive/My Drive/bipedal/weights/hardcore/actor.pth",
                map_location=self.device,
            )
        )
        self.critic.load_state_dict(
            torch.load(
                "/content/drive/My Drive/bipedal/weights/hardcore/critic.pth",
                map_location=self.device,
            )
        )
        self.actor_target.load_state_dict(
            torch.load(
                "/content/drive/My Drive/bipedal/weights/hardcore/actor_t.pth",
                map_location=self.device,
            )
        )
        self.critic_target.load_state_dict(
            torch.load(
                "/content/drive/My Drive/bipedal/weights/hardcore/critic_t.pth",
                map_location=self.device,
            )
        )
        self.sysmodel.load_state_dict(
            torch.load(
                "/content/drive/My Drive/bipedal/weights/hardcore/sysmodel.pth",
                map_location=self.device,
            )
        )

    def add_to_replay_memory(self, transition, buffername):
        # add samples to replay memory
        buffername.append(transition)

    def get_random_sample_from_replay_mem(self, buffername):
        # random samples from replay memory
        random_sample = random.sample(buffername, self.batch_size)
        return random_sample

    def learn_and_update_weights_by_replay(self, training_iterations, weight, totrain):
        """Update policy and value parameters using given batch of experience tuples.
        where:
            actor_target(state) -> action
            critic_target(state, action) -> Q-value
        """
        # print(len(self.replay_memory_buffer))
        if len(self.replay_memory_buffer) < 1e4:
            return 1
        for it in range(training_iterations):
            mini_batch = self.get_random_sample_from_replay_mem(
                self.replay_memory_buffer
            )
            state_batch = (
                torch.from_numpy(np.vstack([i[0] for i in mini_batch]))
                .float()
                .to(self.device)
            )
            action_batch = (
                torch.from_numpy(np.vstack([i[1] for i in mini_batch]))
                .float()
                .to(self.device)
            )
            reward_batch = (
                torch.from_numpy(np.vstack([i[2] for i in mini_batch]))
                .float()
                .to(self.device)
            )
            add_reward_batch = (
                torch.from_numpy(np.vstack([i[3] for i in mini_batch]))
                .float()
                .to(self.device)
            )
            next_state_batch = (
                torch.from_numpy(np.vstack([i[4] for i in mini_batch]))
                .float()
                .to(self.device)
            )
            done_list = (
                torch.from_numpy(np.vstack([i[5] for i in mini_batch]).astype(np.uint8))
                .float()
                .to(self.device)
            )

            # Training and updating Actor & Critic networks.

            # Train Critic
            target_actions = self.actor_target(next_state_batch)
            offset_noises = (
                torch.FloatTensor(action_batch.shape)
                .data.normal_(0, self.policy_noise)
                .to(self.device)
            )

            # clip noise
            offset_noises = offset_noises.clamp(-self.noise_clip, self.noise_clip)
            target_actions = (target_actions + offset_noises).clamp(
                self.lower_bound, self.upper_bound
            )

            # Compute the target Q value
            Q_targets1, Q_targets2 = self.critic_target(
                next_state_batch, target_actions
            )
            Q_targets = torch.min(Q_targets1, Q_targets2)
            Q_targets = reward_batch + self.gamma * Q_targets * (1 - done_list)

            # Compute current Q estimates
            current_Q1, current_Q2 = self.critic(state_batch, action_batch)
            # Compute critic loss
            critic_loss = F.mse_loss(current_Q1, Q_targets.detach()) + F.mse_loss(
                current_Q2, Q_targets.detach()
            )
            # Optimize the critic
            self.crt_opt.zero_grad()
            critic_loss.backward()
            self.crt_opt.step()

            self.soft_update_target(self.critic, self.critic_target)

            # Train_sysmodel
            predict_next_state = self.sysmodel(state_batch, action_batch) * (
                1 - done_list
            )
            next_state_batch = next_state_batch * (1 - done_list)
            sysmodel_loss = F.mse_loss(predict_next_state, next_state_batch.detach())
            self.sys_opt.zero_grad()
            sysmodel_loss.backward()
            self.sys_opt.step()

            s_flag = 1 if sysmodel_loss.item() < 0.020 else 0

            # Train Actor
            # Delayed policy updates
            if it % self.policy_freq == 0 and totrain == 1:
                actions = self.actor(state_batch)
                actor_loss1, _ = self.critic_target(state_batch, actions)
                actor_loss1 = actor_loss1.mean()
                actor_loss = -actor_loss1

                if s_flag == 1:
                    p_actions = self.actor(state_batch)
                    p_next_state = self.sysmodel(state_batch, p_actions).clamp(
                        self.obs_lower_bound, self.obs_upper_bound
                    )

                    p_actions2 = self.actor(p_next_state.detach()) * self.upper_bound
                    actor_loss2, _ = self.critic_target(
                        p_next_state.detach(), p_actions2
                    )
                    actor_loss2 = actor_loss2.mean()

                    p_next_state2 = self.sysmodel(
                        p_next_state.detach(), p_actions2
                    ).clamp(self.obs_lower_bound, self.obs_upper_bound)
                    p_actions3 = self.actor(p_next_state2.detach()) * self.upper_bound
                    actor_loss3, _ = self.critic_target(
                        p_next_state2.detach(), p_actions3
                    )
                    actor_loss3 = actor_loss3.mean()

                    actor_loss_final = (
                        actor_loss - weight * (actor_loss2) - 0.5 * weight * actor_loss3
                    )
                else:
                    actor_loss_final = actor_loss

                self.act_opt.zero_grad()
                actor_loss_final.backward()
                self.act_opt.step()

                # Soft update target models

                self.soft_update_target(self.actor, self.actor_target)

        return sysmodel_loss.item()

    def soft_update_target(self, local_model, target_model):
        """Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target
        Params
        ======
            local_model: PyTorch model (weights will be copied from)
            target_model: PyTorch model (weights will be copied to)
            tau (float): interpolation parameter
        """
        for target_param, local_param in zip(
            target_model.parameters(), local_model.parameters()
        ):
            target_param.data.copy_(
                self.tau * local_param.data + (1.0 - self.tau) * target_param.data
            )

    def policy(self, state):
        """select action based on ACTOR"""
        state = torch.from_numpy(state).float().unsqueeze(0).to(self.device)
        self.actor.eval()
        with torch.no_grad():
            actions = self.actor(state).cpu().data.numpy()
        self.actor.train()
        # Adding noise to action
        shift_action = np.random.normal(
            0, self.std_noise, size=self.env.action_space.shape[0]
        )
        sampled_actions = actions + shift_action
        # We make sure action is within bounds
        legal_action = np.clip(sampled_actions, self.lower_bound, self.upper_bound)
        return np.squeeze(legal_action)

    def select_action(self, state):
        """select action based on ACTOR"""
        state = torch.from_numpy(state).float().unsqueeze(0).to(self.device)
        with torch.no_grad():
            actions = self.actor_target(state).cpu().data.numpy()
        return np.squeeze(actions)

    def eval_policy(self, env_name, seed, eval_episodes):
        eval_env = env_name
        eval_env.seed(seed + 100)

        avg_reward = 0.0
        for _ in range(eval_episodes):
            state, done = eval_env.reset(), False
            while not done:
                action = self.select_action(np.array(state))
                state, reward, done, _ = eval_env.step(action)
                avg_reward += reward
        avg_reward /= eval_episodes

        print("---------------------------------------")
        print(f"Evaluation over {eval_episodes} episodes: {avg_reward:.3f}")
        print("---------------------------------------")
        return avg_reward