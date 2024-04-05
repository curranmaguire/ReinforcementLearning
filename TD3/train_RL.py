import copy
import os
import gym
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
import sys

from TD3 import TD3, ExperienceReplay

print("----------Starting Training----------")

device = (
    torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
)

plot_interval = 10  # update the plot every N episodes
video_every = (
    100  # videos can take a very long time to render so only do it every N episodes
)

env = gym.make("BipedalWalkerHardcore-v3")
# env = gym.make("BipedalWalkerHardcore-v3") # only attempt this when your agent has solved BipedalWalker-v3
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
log_f = open("agent-log.txt", "w+")

# variables for td3
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.shape[0]
max_action = float(env.action_space.high[0])
buffer_size = 1000000
batch_size = 100
noise = 0.1
buffer = ExperienceReplay(buffer_size, batch_size, device)

# initialise agent
agent = TD3(state_dim, action_dim, max_action, env, device)
max_episodes = 10000
max_timesteps = 2000

# training procedure:
for episode in range(1, max_episodes + 1):
    ep_reward = 0
    state = env.reset()
    for t in range(max_timesteps):

        # select the agent action + add noise
        action = agent.select_action(state) + np.random.normal(
            0, max_action * noise, size=action_dim
        )
        action = action.clip(env.action_space.low, env.action_space.high)

        # take action in environment and get r and s'
        next_state, reward, done, _ = env.step(action)
        buffer.store_transition(state, action, reward, next_state, done)
        state = next_state

        env.render()
        if len(buffer) > batch_size:
            agent.train(buffer, t)

        ep_reward += reward

        # stop iterating when the episode finished
        if done or t == (max_timesteps - 1):
            print("Episode ", episode, " finished with reward: ", ep_reward)
            print("Finished at timestep ", t)
            break

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
        plt.savefig(f"{os.getcwd()}/TD3/plots/{episode}_episode_plot.png")
        plt.close()
