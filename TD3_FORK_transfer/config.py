import torch


class config:
    def __init__(self):
        self.max_episodes = 3700
        self.seed = 42
        self.video_every = 100  # videos can take a very long time to render so only do it every N episodes
        self.plot_interval = 10
        self.batch_size = 500
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        if torch.backends.mps.is_available():
            self.device = torch.device("mps")
        else:
            self.device = torch.device("cpu")
        self.buffer_size = 1000000
        
        self.noise = 0.1
        self.max_timesteps = 2000
        self.easy_ep_num = 100
        self.action_size = 0
        self.state_size = 0
        self.max_action = 0
        self.min_state = 0
        self.max_state = 0
        self.hidden_dim = 256
        self.lr = 3e-4
        self.update_after = 50
        self.update_every = 1
        self.explore_duration = 200
        self.test_steps = 10
        self.sys_weight = 0.5
        self.sys_weight2 = 0.4
        self.sys_threshold = 0.020
