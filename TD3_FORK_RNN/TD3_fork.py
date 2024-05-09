# https://www.researchgate.net/publication/320296763_Recurrent_Network-based_Deterministic_Policy_Gradient_for_Solving_Bipedal_Walking_Challenge_on_Rugged_Terrains
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import gym
import random
from torch.distributions import Normal

# https://github.com/ugurcanozalp/td3-sac-bipedal-walker-hardcore-v3/blob/main/td3_agent.py
# https://github.com/vy007vikas/PyTorch-ActorCriticRL
# https://github.com/honghaow/FORK/blob/master/TD3-FORK/TD3_FORK.py
EPS = 0.003


class MyLSTM(nn.Module):
    def __init__(self, input_size, hidden_size=96, batch_first=True):
        super(MyLSTM, self).__init__()
        self.lstm = nn.LSTM(
            input_size,
            hidden_size=hidden_size,
            batch_first=batch_first,
            bidirectional=False,
            num_layers=2,
            dropout=0,
        )
        self.lstm.bias_hh_l0.data.fill_(
            -0.2
        )  # force lstm to output to depend more on last state at the initialization.

    def forward(self, observations, hidden=None):
        summary, (hidden1, hidden2) = self.lstm(observations, hidden)
        return summary[:, -1], (hidden1, hidden2)


class Critic(nn.Module):

    def __init__(self, state_dim=24, action_dim=4):
        """
        :param state_dim: Dimension of input state (int)
        :param action_dim: Dimension of input action (int)
        :return:
        """
        super(Critic, self).__init__()

        self.state_dim = state_dim
        self.action_dim = action_dim

        self.state_encoder = MyLSTM(
            input_size=self.state_dim, hidden_size=96, batch_first=True
        )

        self.fc2 = nn.Linear(96 + self.action_dim, 192)
        nn.init.xavier_uniform_(self.fc2.weight, gain=nn.init.calculate_gain("tanh"))

        self.fc_out = nn.Linear(192, 1, bias=False)
        nn.init.uniform_(self.fc_out.weight, -0.003, +0.003)

        self.act = nn.Tanh()

    def forward(self, state, action):
        """
        returns Value function Q(s,a) obtained from critic network
        :param state: Input state (Torch Variable : [n,state_dim] )
        :param action: Input Action (Torch Variable : [n,action_dim] )
        :return: Value function : Q(S,a) (Torch Variable : [n,1] )
        """
        if state.dim() == 2:
            state = state.unsqueeze(1)  # Add sequence length dimension
        s, (y, z) = self.state_encoder(state)  # Unpack the tuple correctly

        x = torch.cat((s, action), dim=1)
        x = self.act(self.fc2(x))
        x = self.fc_out(x) * 10
        return x


class SysModel(nn.Module):

    def __init__(self, state_dim=24, action_dim=4):
        """
        :param state_dim: Dimension of input state (int)
        :param action_dim: Dimension of input action (int)
        :return:
        """
        super(SysModel, self).__init__()

        self.state_dim = state_dim
        self.action_dim = action_dim

        self.state_encoder = MyLSTM(
            input_size=self.state_dim, hidden_size=96, batch_first=True
        )

        self.fc2 = nn.Linear(96 + self.action_dim, 192)
        nn.init.xavier_uniform_(self.fc2.weight, gain=nn.init.calculate_gain("tanh"))

        self.fc_out = nn.Linear(192, state_dim, bias=False)
        nn.init.uniform_(self.fc_out.weight, -0.003, +0.003)

        self.act = nn.Tanh()

    def forward(self, state, action):
        """
        returns Value function Q(s,a) obtained from critic network
        :param state: Input state (Torch Variable : [n,state_dim] )
        :param action: Input Action (Torch Variable : [n,action_dim] )
        :return: Value function : Q(S,a) (Torch Variable : [n,1] )
        """
        if state.dim() == 2:
            state = state.unsqueeze(1)  # Add sequence length dimension
        s, (y, z) = self.state_encoder(state)  # Unpack the tuple correctly

        x = torch.cat((s, action), dim=1)
        x = self.act(self.fc2(x))
        x = self.fc_out(x)
        return x


class Actor(nn.Module):

    def __init__(self, state_dim=24, action_dim=4, stochastic=False):
        """
        :param state_dim: Dimension of input state (int)
        :param action_dim: Dimension of output action (int)
        :param action_lim: Used to limit action in [-action_lim,action_lim]
        :return:
        """
        super(Actor, self).__init__()

        self.state_dim = state_dim
        self.action_dim = action_dim
        self.stochastic = stochastic

        self.state_encoder = MyLSTM(
            input_size=self.state_dim, hidden_size=96, batch_first=True
        )

        self.fc = nn.Linear(96, action_dim, bias=False)
        nn.init.uniform_(self.fc.weight, -0.003, +0.003)
        # nn.init.zeros_(self.fc.bias)

        if self.stochastic:
            self.log_std = nn.Linear(96, action_dim, bias=False)
            nn.init.uniform_(self.log_std.weight, -0.003, +0.003)
            # nn.init.zeros_(self.log_std.bias)

        self.tanh = nn.Tanh()

    def forward(self, state, explore=True):
        """
        returns either:
        - deterministic policy function mu(s) as policy action.
        - stochastic action sampled from tanh-gaussian policy, with its entropy value.
        this function returns actions lying in (-1,1)
        :param state: Input state (Torch Variable : [n,state_dim] )
        :return: Output action (Torch Variable: [n,action_dim] )
        """
        if state.dim() == 2:
            state = state.unsqueeze(1)  # Add sequence length dimension
        s, _ = self.state_encoder(state)  # Unpack the tuple correctly

        if self.stochastic:
            means = self.fc(s)
            log_stds = self.log_std(s)
            log_stds = torch.clamp(log_stds, min=-10.0, max=2.0)
            stds = log_stds.exp()
            dists = Normal(means, stds)
            if explore:
                x = dists.rsample()
            else:
                x = means
            actions = self.tanh(x)
            log_probs = dists.log_prob(x) - torch.log(1 - actions.pow(2) + 1e-6)
            entropies = -log_probs.sum(dim=1, keepdim=True)
            return actions, entropies
        else:
            actions = self.tanh(self.fc(s))
            return actions


class Sys_R(nn.Module):

    def __init__(self, state_dim=24, action_dim=4):
        """
        :param state_dim: Dimension of input state (int)
        :param action_dim: Dimension of input action (int)
        :return:
        """
        super(Sys_R, self).__init__()

        self.state_dim = state_dim
        self.action_dim = action_dim

        self.state_encoder = MyLSTM(
            input_size=self.state_dim, hidden_size=96, batch_first=True
        )

        self.fc2 = nn.Linear(96 + 96 + self.action_dim, 192)
        nn.init.xavier_uniform_(self.fc2.weight, gain=nn.init.calculate_gain("tanh"))

        self.fc_out = nn.Linear(192, 1, bias=False)
        nn.init.uniform_(self.fc_out.weight, -0.003, +0.003)

        self.act = nn.Tanh()

    def forward(self, state, next_state, action):
        """
        returns Value function Q(s,a) obtained from critic network
        :param state: Input state (Torch Variable : [n,state_dim] )
        :param action: Input Action (Torch Variable : [n,action_dim] )
        :return: Value function : Q(S,a) (Torch Variable : [n,1] )
        """
        if state.dim() == 2:
            state = state.unsqueeze(1)
        if next_state.dim() == 2:
            next_state = next_state.unsqueeze(1)
        s, _ = self.state_encoder(state)
        ns, _ = self.state_encoder(next_state)

        x = torch.cat([s, ns, action], dim=1)
        x = self.act(self.fc2(x))
        x = self.fc_out(x) * 10
        return x


import torch
from torch import optim
import numpy as np
import os
from noise import (
    OrnsteinUhlenbeckNoise,
    DecayingOrnsteinUhlenbeckNoise,
    GaussianNoise,
    DecayingGaussianNoise,
    DecayingRandomNoise,
)
from itertools import chain
from utils import ReplayBuffer


class TD3Agent:
    rl_type = "td3"

    def __init__(
        self,
        config,
        clip_low,
        clip_high,
        update_freq=int(2),
        lr=4e-4,
        weight_decay=0,
        gamma=0.98,
        tau=0.01,
        buffer_size=int(500000),
    ):

        self.state_size = config.state_size
        self.action_size = config.action_size
        self.update_freq = update_freq
        self.obs_lower_bound = config.min_state
        self.obs_upper_bound = config.max_state
        self.learn_call = int(0)
        self.update_sys = 0
        self.gamma = gamma
        self.tau = tau
        self.batch_size = config.batch_size

        self.device = config.device

        self.clip_low = torch.tensor(clip_low).to(self.device)
        self.clip_high = torch.tensor(clip_high).to(self.device)

        self.train_actor = Actor().to(self.device)
        self.target_actor = Actor().to(self.device).eval()
        self.hard_update(
            self.train_actor, self.target_actor
        )  # hard update at the beginning
        self.actor_optim = torch.optim.AdamW(
            self.train_actor.parameters(),
            lr=lr,
            weight_decay=weight_decay,
            amsgrad=True,
        )
        print(
            f"Number of paramters of Actor Net: {sum(p.numel() for p in self.train_actor.parameters())}"
        )

        self.train_critic_1 = Critic().to(self.device)
        self.target_critic_1 = Critic().to(self.device).eval()
        self.hard_update(
            self.train_critic_1, self.target_critic_1
        )  # hard update at the beginning
        self.critic_1_optim = torch.optim.AdamW(
            self.train_critic_1.parameters(),
            lr=lr,
            weight_decay=weight_decay,
            amsgrad=True,
        )

        self.train_critic_2 = Critic().to(self.device)
        self.target_critic_2 = Critic().to(self.device).eval()
        self.hard_update(
            self.train_critic_2, self.target_critic_2
        )  # hard update at the beginning
        self.critic_2_optim = torch.optim.AdamW(
            self.train_critic_2.parameters(),
            lr=lr,
            weight_decay=weight_decay,
            amsgrad=True,
        )
        print(
            f"Number of paramters of Single Critic Net: {sum(p.numel() for p in self.train_critic_2.parameters())}"
        )

        self.sysmodel = SysModel().to(self.device)
        self.sysmodel_optimizer = torch.optim.Adam(self.sysmodel.parameters(), lr=3e-4)

        self.sysr = Sys_R().to(self.device)
        self.sysr_optimizer = torch.optim.Adam(self.sysr.parameters(), lr=3e-4)

        self.noise_generator = DecayingOrnsteinUhlenbeckNoise(
            mu=np.zeros(config.action_size),
            theta=4.0,
            sigma=1.2,
            dt=0.04,
            sigma_decay=0.9995,
        )
        self.target_noise = GaussianNoise(
            mu=np.zeros(config.action_size), sigma=0.2, clip=0.4
        )

        self.memory = ReplayBuffer(
            action_size=config.action_size,
            buffer_size=buffer_size,
            batch_size=self.batch_size,
            device=self.device,
        )
        self.sysmodel_loss = 0
        self.sysr_loss = 0
        self.sys_weight = config.sys_weight
        self.sys_weight2 = config.sys_weight2
        self.sys_threshold = config.sys_threshold
        self.mse_loss = torch.nn.MSELoss()

    def learn_with_batches(self, state, action, reward, next_state, done):
        self.memory.add(state, action, reward, next_state, done)
        self.learn_one_step()

    def learn_one_step(self):
        if len(self.memory) > self.batch_size:
            exp = self.memory.sample()
            self.learn(exp)

    def learn(self, exp):
        self.learn_call += 1
        states, actions, rewards, next_states, done = exp

        # update critic
        with torch.no_grad():
            next_actions = self.target_actor(next_states)
            next_actions = next_actions + torch.from_numpy(
                self.target_noise()
            ).float().to(self.device)
            next_actions = torch.clamp(next_actions, self.clip_low, self.clip_high)
            Q_targets_next_1 = self.target_critic_1(next_states, next_actions)
            Q_targets_next_2 = self.target_critic_2(next_states, next_actions)
            Q_targets_next = torch.min(Q_targets_next_1, Q_targets_next_2).detach()
            Q_targets = rewards + (self.gamma * Q_targets_next * (1 - done))
            # Q_targets = rewards + (self.gamma * Q_targets_next)

        # train the critic
        Q_expected_1 = self.train_critic_1(states, actions)
        critic_1_loss = self.mse_loss(Q_expected_1, Q_targets)
        # critic_1_loss = torch.nn.SmoothL1Loss()(Q_expected_1, Q_targets)

        self.critic_1_optim.zero_grad(set_to_none=True)
        critic_1_loss.backward()
        # torch.nn.utils.clip_grad_norm_(self.train_critic_1.parameters(), 1)
        self.critic_1_optim.step()

        Q_expected_2 = self.train_critic_2(states, actions)
        critic_2_loss = self.mse_loss(Q_expected_2, Q_targets)
        # critic_2_loss = torch.nn.SmoothL1Loss()(Q_expected_2, Q_targets)

        self.critic_2_optim.zero_grad(set_to_none=True)
        critic_2_loss.backward()
        # torch.nn.utils.clip_grad_norm_(self.train_critic_2.parameters(), 1)
        self.critic_2_optim.step()

        predict_next_state = self.sysmodel(states, actions)
        predict_next_state = predict_next_state.clamp(
            self.obs_lower_bound, self.obs_upper_bound
        )
        sysmodel_loss = F.smooth_l1_loss(predict_next_state, next_states.detach())

        self.sysmodel_optimizer.zero_grad()
        sysmodel_loss.backward()
        self.sysmodel_optimizer.step()
        self.sysmodel_loss = sysmodel_loss.item()

        predict_reward = self.sysr(states, next_states, actions)
        sysr_loss = F.mse_loss(predict_reward, rewards.detach())
        self.sysr_optimizer.zero_grad()
        sysr_loss.backward()
        self.sysr_optimizer.step()
        self.sysr_loss = sysr_loss.item()
        s_flag = 1 if sysmodel_loss.item() < self.sys_threshold else 0

        if self.learn_call % self.update_freq == 0:
            self.learn_call = 0
            # update actor
            actions_pred = self.train_actor(states)
            actor_loss1 = -self.train_critic_1(states, actions_pred).mean()
            if s_flag == 1:
                p_next_state = self.sysmodel(states, self.train_actor(states))
                p_next_state = p_next_state.clamp(
                    self.obs_lower_bound, self.obs_upper_bound
                )
                actions2 = self.train_actor(p_next_state.detach())
                p_next_r = self.sysr(
                    states, p_next_state.detach(), self.train_actor(states)
                )
                p_next_state2 = self.sysmodel(
                    p_next_state, self.train_actor(p_next_state.detach())
                )
                p_next_state2 = p_next_state2.clamp(
                    self.obs_lower_bound, self.obs_upper_bound
                )
                p_next_r2 = self.sysr(
                    p_next_state.detach(),
                    p_next_state2.detach(),
                    self.train_actor(p_next_state.detach()),
                )
                actions3 = self.train_actor(p_next_state2.detach())

                actor_loss2 = self.train_critic_1(p_next_state2.detach(), actions3)
                actor_loss3 = -(
                    p_next_r + self.gamma * p_next_r2 + self.gamma**2 * actor_loss2
                ).mean()
                actor_loss = actor_loss1 + self.sys_weight * actor_loss3
                self.update_sys += 1
            else:

                actor_loss = actor_loss1

            # Optimize the train_actor
            self.critic_1_optim.zero_grad()
            self.sysmodel_optimizer.zero_grad()
            self.actor_optim.zero_grad()
            actor_loss.backward()
            self.actor_optim.step()

            # using soft upates
            self.soft_update(self.train_actor, self.target_actor)
            self.soft_update(self.train_critic_1, self.target_critic_1)
            self.soft_update(self.train_critic_2, self.target_critic_2)

    @torch.no_grad()
    def get_action(self, state, explore=False):
        # self.train_actor.eval()
        state = torch.from_numpy(state).unsqueeze(0).float().to(self.device)
        # with torch.no_grad():
        action = self.train_actor(state).cpu().data.numpy()[0]
        # self.train_actor.train()

        if explore:
            noise = self.noise_generator()
            # print(noise)
            action += noise
        return action

    def soft_update(self, local_model, target_model):
        for target_param, local_param in zip(
            target_model.parameters(), local_model.parameters()
        ):
            target_param.data.copy_(
                self.tau * local_param.data + (1.0 - self.tau) * target_param.data
            )

    def hard_update(self, local_model, target_model):
        for target_param, local_param in zip(
            target_model.parameters(), local_model.parameters()
        ):
            target_param.data.copy_(local_param.data)

    def save_ckpt(self, model_type, env_type, prefix="last"):
        actor_file = os.path.join(
            "models",
            self.rl_type,
            env_type,
            "_".join([prefix, model_type, "actor.pth"]),
        )
        critic_1_file = os.path.join(
            "models",
            self.rl_type,
            env_type,
            "_".join([prefix, model_type, "critic_1.pth"]),
        )
        critic_2_file = os.path.join(
            "models",
            self.rl_type,
            env_type,
            "_".join([prefix, model_type, "critic_2.pth"]),
        )
        torch.save(self.train_actor.state_dict(), actor_file)
        torch.save(self.train_critic_1.state_dict(), critic_1_file)
        torch.save(self.train_critic_2.state_dict(), critic_2_file)

    def load_ckpt(self, model_type, env_type, prefix="last"):
        actor_file = os.path.join(
            "models",
            self.rl_type,
            env_type,
            "_".join([prefix, model_type, "actor.pth"]),
        )
        critic_1_file = os.path.join(
            "models",
            self.rl_type,
            env_type,
            "_".join([prefix, model_type, "critic_1.pth"]),
        )
        critic_2_file = os.path.join(
            "models",
            self.rl_type,
            env_type,
            "_".join([prefix, model_type, "critic_2.pth"]),
        )
        try:
            self.train_actor.load_state_dict(
                torch.load(actor_file, map_location=self.device)
            )
        except:
            print("Actor checkpoint cannot be loaded.")
        try:
            self.train_critic_1.load_state_dict(
                torch.load(critic_1_file, map_location=self.device)
            )
            self.train_critic_2.load_state_dict(
                torch.load(critic_2_file, map_location=self.device)
            )
        except:
            print("Critic checkpoints cannot be loaded.")

    def train_mode(self):
        self.train_actor.train()
        self.train_critic_1.train()
        self.train_critic_2.train()

    def eval_mode(self):
        self.train_actor.eval()
        self.train_critic_1.eval()
        self.train_critic_2.eval()

    def freeze_networks(self):
        for p in chain(
            self.train_actor.parameters(),
            self.train_critic_1.parameters(),
            self.train_critic_2.parameters(),
        ):
            p.requires_grad = False

    def step_end(self):
        self.noise_generator.step_end()

    def episode_end(self):
        self.noise_generator.episode_end()


# https://github.com/ikoself.config.devicerch-a2c-ppo-acktr-gail/blob/master/a2c_ppo_acktr/algo/a2c_acktr.py
# https://github.com/zhihanyang2022/off-policy-continuous-control/blob/pub/offpcc/algorithms_recurrent/recurrent_td3.py
'''import numpy as np
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
from copy import deepcopy
from utils import RecurrentReplayBuffer, RecurrentBatch
from config import config


def set_requires_grad_flag(net: nn.Module, requires_grad: bool) -> None:
    for p in net.parameters():
        p.requires_grad = requires_grad


def create_target(net: nn.Module) -> nn.Module:
    target = deepcopy(net)
    set_requires_grad_flag(target, False)
    return target


def polyak_update(targ_net: nn.Module, pred_net: nn.Module, polyak: float) -> None:
    with torch.no_grad():  # no grad is not actually required here; only for sanity check
        for targ_p, p in zip(targ_net.parameters(), pred_net.parameters()):
            targ_p.data.copy_(targ_p.data * polyak + p.data * (1 - polyak))


def mean_of_unmasked_elements(tensor: torch.tensor, mask: torch.tensor) -> torch.tensor:
    return torch.mean(tensor * mask) / mask.sum() * np.prod(mask.shape)


class Summarizer(nn.Module):
    def __init__(self, config, num_layers=2, recurrent_type="lstm"):
        super().__init__()
        if recurrent_type == "lstm":
            self.rnn = nn.LSTM(
                config.state_size,
                config.hidden_dim,
                batch_first=True,
                num_layers=num_layers,
            )
        elif recurrent_type == "rnn":
            self.rnn = nn.RNN(
                config.state_size,
                config.hidden_dim,
                batch_first=True,
                num_layers=num_layers,
            )
        elif recurrent_type == "gru":
            self.rnn = nn.GRU(
                config.state_size,
                config.hidden_dim,
                batch_first=True,
                num_layers=num_layers,
            )
        else:
            assert f"{recurrent_type} not recognized"

    def forward(self, hidden=None, return_hidden=False):
        self.rnn.flatten_parameters()
        summary, hidden = self.rnn(self.config.devices, hidden)
        if return_hidden:
            return summary, hidden
        else:
            return summary


# Actor Neural Network
class Actor(nn.Module):
    def __init__(self, state_size, config.action_size, seed, fc_units=400, fc1_units=300):
        super(Actor, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.fc1 = nn.Linear(state_size, fc_units)
        self.fc2 = nn.Linear(fc_units, fc1_units)
        self.fc3 = nn.Linear(fc1_units, config.action_size)

    def forward(self, state):
        """Build an actor (policy) network that maps states -> actions."""
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        return F.torch.tanh(self.fc3(x))


# Q1-Q2-Critic Neural Network


class Critic(nn.Module):
    def __init__(self, state_size, config.action_size, seed, fc1_units=400, fc2_units=300):
        super(Critic, self).__init__()
        self.seed = torch.manual_seed(seed)

        # Q1 architecture
        self.l1 = nn.Linear(state_size + config.action_size, fc1_units)
        self.l2 = nn.Linear(fc1_units, fc2_units)
        self.l3 = nn.Linear(fc2_units, 1)

    def forward(self, state, action):
        """Build a critic network that maps (state, action) pairs."""
        xa = torch.cat([state, action], 1)

        x1 = F.relu(self.l1(xa))
        x1 = F.relu(self.l2(x1))
        x1 = self.l3(x1)

        return x1


class SysModel(nn.Module):
    def __init__(self, state_size, config.action_size, fc1_units=400, fc2_units=300):
        super(SysModel, self).__init__()
        self.l1 = nn.Linear(state_size + config.action_size, fc1_units)
        self.l2 = nn.Linear(fc1_units, fc2_units)
        self.l3 = nn.Linear(fc2_units, state_size)

    def forward(self, state, action):
        """Build a system model to predict the next state a eate."""
        xa = torch.cat([state, action], 1)

        x1 = F.relu(self.l1(xa))
        x1 = F.relu(self.l2(x1))
        x1 = self.l3(x1)

        return x1


class TD3_FORK:
    def __init__(
        self,
        config,
        hidden_dim=256,
        gamma=0.99,
        lr=3e-4,
        polyak=0.995,
        action_noise=0.1,  # standard deviation of action noise
        target_noise=0.2,  # standard deviation of target smoothing noise
        noise_clip=0.5,  # max abs value of target smoothing noise
        policy_delay=2,
    ):

        # hyper-parameters

        self.input_dim = config.state_size
        self.action_dim = config.config.action_size
        self.hidden_dim = hidden_dim
        self.gamma = gamma
        self.lr = config.lr
        self.polyak = polyak

        self.action_noise = action_noise
        self.target_noise = target_noise
        self.noise_clip = noise_clip

        self.policy_delay = policy_delay

        # trackers

        self.hidden = None
        self.num_Q_updates = 0
        self.mean_Q1_value = 0

        self.actor_summarizer = Summarizer(config).to(config.device)

        self.Q1_summarizer = Summarizer(config).to(config.device)

        self.Q2_summarizer = Summarizer(config).to(config.device)

        self.actor = Actor(config.state_size, config.config.action_size, config.seed).to(
            config.device
        )

        self.Q1 = Critic(config.state_size, config.config.action_size, config.seed).to(
            config.device
        )

        self.Q2 = Critic(config.state_size, config.config.action_size, config.seed).to(
            config.device
        )

        self.sys_model = SysModel(config.state_size, config.config.action_size).to(
            config.device
        )
        # optimizers)

        self.actor_summarizer_optimizer = optim.Adam(
            self.actor_summarizer.parameters(), lr=config.lr
        )
        self.Q1_summarizer_optimizer = optim.Adam(
            self.Q1_summarizer.parameters(), lr=config.lr
        )
        self.Q2_summarizer_optimizer = optim.Adam(
            self.Q2_summarizer.parameters(), lr=config.lr
        )

        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=config.lr)
        self.Q1_optimizer = optim.Adam(self.Q1.parameters(), lr=config.lr)
        self.Q2_optimizer = optim.Adam(self.Q2.parameters(), lr=config.lr)

    def reinitialize_hidden(self) -> None:
        self.hidden = None

    def act(self, observation: np.array, deterministic: bool) -> np.array:
        with torch.no_grad():
            observation = (
                torch.tensor(observation)
                .unsqueeze(0)
                .unsqueeze(0)
                .float()
                .to(self.config.device)
            )
            summary, self.hidden = self.actor_summarizer(
                observation, self.hidden, return_hidden=True
            )
            greedy_action = (
                self.actor(summary).view(-1).cpu().numpy()
            )  # view as 1d -> to cpu -> to numpy
            if deterministic:
                return greedy_action
            else:
                return np.clip(
                    greedy_action
                    + self.action_noise * np.random.randn(self.action_dim),
                    -1.0,
                    1.0,
                )

    def update_networks(self, b: RecurrentBatch):

        bs, num_bptt = b.r.shape[0], b.r.shape[1]

        # compute summary

        actor_summary = self.actor_summarizer(b.o)
        Q1_summary = self.Q1_summarizer(b.o)
        Q2_summary = self.Q2_summarizer(b.o)

        actor_summary_targ = self.actor_summarizer_targ(b.o)
        Q1_summary_targ = self.Q1_summarizer_targ(b.o)
        Q2_summary_targ = self.Q2_summarizer_targ(b.o)

        actor_summary_1_T, actor_summary_2_Tplus1 = (
            actor_summary[:, :-1, :],
            actor_summary_targ[:, 1:, :],
        )
        Q1_summary_1_T, Q1_summary_2_Tplus1 = (
            Q1_summary[:, :-1, :],
            Q1_summary_targ[:, 1:, :],
        )
        Q2_summary_1_T, Q2_summary_2_Tplus1 = (
            Q2_summary[:, :-1, :],
            Q2_summary_targ[:, 1:, :],
        )

        assert actor_summary.shape == (bs, num_bptt + 1, self.hidden_dim)

        # compute predictions

        Q1_predictions = self.Q1(Q1_summary_1_T, b.a)
        Q2_predictions = self.Q2(Q2_summary_1_T, b.a)

        assert Q1_predictions.shape == (bs, num_bptt, 1)
        assert Q2_predictions.shape == (bs, num_bptt, 1)

        # compute targets

        with torch.no_grad():

            na = self.actor_targ(actor_summary_2_Tplus1)
            noise = torch.clamp(
                torch.randn(na.size()) * self.target_noise,
                -self.noise_clip,
                self.noise_clip,
            ).to(self.config.device)
            smoothed_na = torch.clamp(na + noise, -1, 1)

            n_min_Q_targ = torch.min(
                self.Q1_targ(Q1_summary_2_Tplus1, smoothed_na),
                self.Q2_targ(Q2_summary_2_Tplus1, smoothed_na),
            )

            targets = b.r + self.gamma * (1 - b.d) * n_min_Q_targ

            assert na.shape == (bs, num_bptt, self.action_dim)
            assert n_min_Q_targ.shape == (bs, num_bptt, 1)
            assert targets.shape == (bs, num_bptt, 1)

        # compute td error

        Q1_loss_elementwise = (Q1_predictions - targets) ** 2
        Q1_loss = mean_of_unmasked_elements(Q1_loss_elementwise, b.m)

        Q2_loss_elementwise = (Q2_predictions - targets) ** 2
        Q2_loss = mean_of_unmasked_elements(Q2_loss_elementwise, b.m)

        assert Q1_loss.shape == ()
        assert Q2_loss.shape == ()

        # reduce td error

        self.Q1_summarizer_optimizer.zero_grad()
        self.Q1_optimizer.zero_grad()
        Q1_loss.backward()
        self.Q1_summarizer_optimizer.step()
        self.Q1_optimizer.step()

        self.Q2_summarizer_optimizer.zero_grad()
        self.Q2_optimizer.zero_grad()
        Q2_loss.backward()
        self.Q2_summarizer_optimizer.step()
        self.Q2_optimizer.step()

        self.num_Q_updates += 1

        if (
            self.num_Q_updates % self.policy_delay == 0
        ):  # delayed policy update; special in TD3

            # compute policy loss

            a = self.actor(actor_summary_1_T)
            Q1_values = self.Q1(Q1_summary_1_T.detach(), a)  # val stands for values
            policy_loss_elementwise = -Q1_values
            policy_loss = mean_of_unmasked_elements(policy_loss_elementwise, b.m)

            self.mean_Q1_value = float(-policy_loss)
            assert a.shape == (bs, num_bptt, self.action_dim)
            assert Q1_values.shape == (bs, num_bptt, 1)
            assert policy_loss.shape == ()

            # reduce policy loss

            self.actor_summarizer_optimizer.zero_grad()
            self.actor_optimizer.zero_grad()
            policy_loss.backward()
            self.actor_summarizer_optimizer.step()
            self.actor_optimizer.step()

            # update target networks

            polyak_update(
                targ_net=self.actor_summarizer_targ,
                pred_net=self.actor_summarizer,
                polyak=self.polyak,
            )
            polyak_update(
                targ_net=self.Q1_summarizer_targ,
                pred_net=self.Q1_summarizer,
                polyak=self.polyak,
            )
            polyak_update(
                targ_net=self.Q2_summarizer_targ,
                pred_net=self.Q2_summarizer,
                polyak=self.polyak,
            )

            polyak_update(
                targ_net=self.actor_targ, pred_net=self.actor, polyak=self.polyak
            )
            polyak_update(targ_net=self.Q1_targ, pred_net=self.Q1, polyak=self.polyak)
            polyak_update(targ_net=self.Q2_targ, pred_net=self.Q2, polyak=self.polyak)

        return {
            # for learning the q functions
            "(qfunc) Q1 pred": float(mean_of_unmasked_elements(Q1_predictions, b.m)),
            "(qfunc) Q2 pred": float(mean_of_unmasked_elements(Q2_predictions, b.m)),
            "(qfunc) Q1 loss": float(Q1_loss),
            "(qfunc) Q2 loss": float(Q2_loss),
            # for learning the actor
            "(actor) Q1 value": self.mean_Q1_value,
        }

    def copy_networks_from(self, algorithm) -> None:

        self.actor_summarizer.load_state_dict(algorithm.actor_summarizer.state_dict())
        self.actor_summarizer_targ.load_state_dict(
            algorithm.actor_summarizer_targ.state_dict()
        )

        self.Q1_summarizer.load_state_dict(algorithm.Q1_summarizer.state_dict())
        self.Q1_summarizer_targ.load_state_dict(
            algorithm.Q1_summarizer_targ.state_dict()
        )

        self.Q2_summarizer.load_state_dict(algorithm.Q2_summarizer.state_dict())
        self.Q2_summarizer_targ.load_state_dict(
            algorithm.Q2_summarizer_targ.state_dict()
        )

        self.actor.load_state_dict(algorithm.actor.state_dict())
        self.actor_targ.load_state_dict(algorithm.actor_targ.state_dict())

        self.Q1.load_state_dict(algorithm.Q1.state_dict())
        self.Q1_targ.load_state_dict(algorithm.Q1_targ.state_dict())

        self.Q2.load_state_dict(algorithm.Q2.state_dict())
        self.Q2_targ.load_state_dict(algorithm.Q2_targ.state_dict())

    def __deepcopy__(self, memo):
        cls = self.__class__
        result = cls.__new__(cls)
        memo[id(self)] = result

        # Copy the attributes manually, excluding the ones that cannot be pickled
        for k, v in self.__dict__.items():
            if k not in [
                "actor",
                "critic",
                "optimizer",
                "device",
            ]:  # Exclude the unpicklable attributes
                setattr(result, k, deepcopy(v, memo))

        # Create new instances of the PyTorch components
        result.actor = Actor(...)  # Create a new actor network
        result.critic = Critic(...)  # Create a new critic network
        result.optimizer = ...  # Create new optimizers

        return result

'''
