import torch


class config:
    def __init__(self):

        self.max_episodes = 5000
        self.batch_size = 100
        self.seed = 42
        self.video_every = 100  # videos can take a very long time to render so only do it every N episodes
        self.plot_interval = 10
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        if torch.backends.mps.is_available():
            self.device = torch.device("mps")
        else:
            self.device = torch.device("cpu")
        self.buffer_size = 1000000
        self.batch_size = 300
        self.noise = 0.1
        self.max_timesteps = 2000
        self.easy_ep_num = 100
        self.action_size = 0
        self.state_size = 0
        self.max_action = 0
        self.hidden_dim = 256
        self.lr = 3e-4
        self.update_after = 50
        self.update_every = 1
        self.explore_duration = 100
        self.test_steps = 100
