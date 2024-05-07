import torch


class config:
    def __init__(self) -> None:

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
