import copy
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
from TD3_fork import TD3_FORK

print("----------Starting Training----------")

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

plot_interval = 10  # update the plot every N episodes
video_every = (
    100  # videos can take a very long time to render so only do it every N episodes
)


env = gym.make("BipedalWalkerHardcore-v3")
agent = TD3_FORK("Bipedalhardcore", env, batch_size=100)
total_episodes = 100000
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
env = gym.wrappers.Monitor(
    env, "./video", video_callable=lambda ep_id: ep_id % video_every == 0, force=True
)

obs_dim = env.observation_space.shape[0]
act_dim = env.action_space.shape[0]

print(
    "The environment has {} observations and the agent can take {} actions".format(
        obs_dim, act_dim
    )
)
print("The device is: {}".format(device))

if device.type != "cpu":
    print("It's recommended to train on the cpu for this")

# in the submission please use seed 42 for verification
seed = 42
torch.manual_seed(seed)
env.seed(seed)
random.seed(seed)
np.random.seed(seed)
env.action_space.seed(seed)

# logging variables
ep_reward = 0
reward_list = []
plot_data = []
log_f = open("TD3_FORK/agent-log.txt", "w+")

# variables for td3
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.shape[0]
max_action = float(env.action_space.high[0])
buffer_size = 1000000
batch_size = 300
noise = 0.1
max_steps = 2000
falling_down = 0
# initialise agent

max_episodes = 1000
max_timesteps = 2000

# training procedure:
for episode in range(1, max_episodes + 1):

    state = env.reset()
    episodic_reward = 0
    timestep = 0
    temp_replay_buffer = []

    for st in range(max_timesteps):

        # select the agent action + add noise
        action = agent.select_action(state) + np.random.normal(
            0, max_action * noise, size=action_dim
        )
        action = action.clip(env.action_space.low, env.action_space.high)

        # take action in environment and get r and s'
        next_state, reward, done, _ = env.step(action)
        ep_reward += reward
        # change the reward to be -5 instead of -100 and 5*reward for the other values
        episodic_reward += reward
        if reward == -100:
            add_reward = -1
            reward = -5
            falling_down += 1
            expcount += 1
        else:
            add_reward = 0
            reward = 5 * reward

        temp_replay_buffer.append((state, action, reward, add_reward, next_state, done))

        state = next_state

        env.render()

        # stop iterating when the episode finished
        if done:
            ep_reward_list.append(episodic_reward)
            if add_reward == -1 or episodic_reward < 250:
                totrain = 1
                for temp in temp_replay_buffer:
                    agent.add_to_replay_memory(temp, agent.replay_memory_buffer)
            elif expcount > 0 and np.random.rand() > 0.5:
                totrain = 1
                expcount -= 10
                for temp in temp_replay_buffer:
                    agent.add_to_replay_memory(temp, agent.replay_memory_buffer)
            break
        timestep += 1
        total_timesteps += 1
    # Training agent only when new experiences are added to the replay buffer
    weight = 1 - np.clip(np.mean(ep_reward_list[-100:]) / 300, 0, 1)
    if totrain == 1:
        sys_loss = agent.learn_and_update_weights_by_replay(timestep, weight, totrain)
    else:
        sys_loss = agent.learn_and_update_weights_by_replay(100, weight, totrain)
    totrain = 0

    # append the episode reward to the reward list
    reward_list.append(ep_reward)

    # do NOT change this logging code - it is used for automated marking!
    log_f.write("episode: {}, reward: {}\n".format(episode, ep_reward))
    log_f.flush()
    ep_reward = 0

    # print reward data every so often - add a graph like this in your report
    if episode % plot_interval == 0:
        plot_data.append(
            [episode, np.array(reward_list).mean(), np.array(reward_list).std()]
        )
        reward_list = []
        # plt.rcParams['figure.dpi'] = 100
        plt.plot(
            [x[0] for x in plot_data], [x[1] for x in plot_data], "-", color="tab:grey"
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
        plt.savefig(f"{os.getcwd()}/TD3_FORK/plots/{episode}_episode_plot.png")
        plt.close()
