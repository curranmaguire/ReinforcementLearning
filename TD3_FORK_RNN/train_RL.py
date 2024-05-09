from copy import deepcopy
import os
import time
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
from TD3_fork import TD3Agent
from config import config
from utils import RecurrentReplayBuffer
from collections import deque

print("----------Starting Training----------")


env = gym.make("BipedalWalkerHardcore-v3")
env_easy = gym.make("BipedalWalker-v3")


# ============================variables
config = config()
start_timestep = 0  # time_step to select action based on Actor
time_start = time.time()  # Init start time
ep_reward_list = []
avg_reward_list = []
total_timesteps = 0
sys_loss = 0
numtrainedexp = 0
save_time = 0
expcount = 0
totrain = 0
# ===========================wrappers
"""env = gym.wrappers.RecordVideo(
    env,
    "./video",
    episode_trigger=lambda ep_id: ep_id % config.video_every == 0,
)
env_easy = gym.wrappers.RecordVideo(
    env_easy,
    "./video_easy",
    episode_trigger=lambda ep_id: ep_id % config.video_every == 0,
)"""

config.state_size = env.observation_space.shape[-1]
config.action_size = env.action_space.shape[-1]
config.max_action = float(env.action_space.high[0])
config.max_state = float(env.observation_space.high[0])
config.min_state = float(env.observation_space.low[0])
# ==========================seeding
torch.manual_seed(config.seed)
env.seed(config.seed)
env_easy.seed(config.seed)
random.seed(config.seed)
np.random.seed(config.seed)
env.action_space.seed(config.seed)
env_easy.action_space.seed(config.seed)
# =========================logging variables
ep_reward = 0
reward_list = []
plot_data = []
log_f = open("agent-log.txt", "w+")

# =========================variables for td3 RNN
episode_len = 0
episode_ret = 0
train_episode_lens = []
train_episode_rets = []
algo_specific_stats_tracker = []

config.max_action = float(env.action_space.high[0])
falling_down = 0


agent = TD3Agent(
    config,
    clip_low=-1,
    clip_high=+1,
)
scores_deque = deque(maxlen=100)
scores = []
test_scores = []
max_score = -np.Inf
print("-----------------beginning training")
# training procedure:
for episode in range(1, config.max_episodes + 1):
    state = env.reset()
    score = 0
    done = False
    agent.train_mode()
    t = int(0)
    while not done and t < config.max_timesteps:
        t += int(1)
        action = agent.get_action(state, explore=True)
        action = action.clip(min=env.action_space.low, max=env.action_space.high)
        next_state, reward, done, info = env.step(action)

        state = next_state
        score += reward
        episode_len += 1
        if reward == -100:
            add_reward = -1
            reward = -5
            falling_down += 1
            expcount += 1
        else:
            add_reward = 0
            reward = 5 * reward

        agent.memory.add(state, action, reward, next_state, done)

        agent.step_end()
        env.render()
    if episode > config.explore_duration:
        agent.episode_end()
        for i in range(t):
            agent.learn_one_step()
    scores_deque.append(score)
    avg_score_100 = np.mean(scores_deque)
    scores.append((episode, score, avg_score_100))
    reward_list.append(score)
    print(
        "\rEpisode {}\tAverage Score: {:.2f}\tScore: {:.2f}".format(
            episode, avg_score_100, score
        ),
        end="",
    )

    if episode % config.plot_interval == 0:
        plot_data.append(
            [episode, np.array(reward_list).mean(), np.array(reward_list).std()]
        )
        reward_list = []
        # plt.rcParams['figure.dpi'] = 100
        plt.plot(
            [x[0] for x in plot_data],
            [x[1] for x in plot_data],
            "-",
            color="tab:grey",
        )
        plt.fill_between(
            [x[0] for x in plot_data],
            [x[1] - x[2] for x in plot_data],
            [x[1] + x[2] for x in plot_data],
            alpha=0.2,
            color="tab:grey",
        )
        plt.xlabel("Episode number")
        plt.ylabel("Episode reward")
        plt.savefig(f"plots/{episode}_episode_plot.png")
        plt.close()
        reward_list.append(ep_reward)
        # do NOT change this logging code - it is used for automated marking!
        log_f.write("episode: {}, reward: {}\n".format(episode, score))
        log_f.flush()
        ep_reward = 0
        total_timesteps += 1


time_end = time.time()
print(f"time taken to train: {time_end-time_start} seconds")
