import numpy as np
import torch
from collections import deque

class MultiAgentHistoryBuffer:
    def __init__(self, num_agents, seq_len, obs_dim, device="cpu"):
        self.num_agents = num_agents
        self.seq_len = seq_len
        self.obs_dim = obs_dim
        self.device = device
        self.buffers = {i: deque(maxlen=seq_len) for i in range(num_agents)}

    def reset(self, initial_obs_list):
        if len(initial_obs_list) != self.num_agents:
            raise ValueError(f"Expected {self.num_agents} initial observations.")
        for i in range(self.num_agents):
            self.buffers[i].clear()
            for _ in range(self.seq_len):
                self.buffers[i].append(initial_obs_list[i])

    def step(self, obs_list):
        for i in range(self.num_agents):
            self.buffers[i].append(obs_list[i])

    def get_sequences_numpy(self):
        sequences = np.array([list(self.buffers[i]) for i in range(self.num_agents)], dtype=np.float32)
        return sequences

    def get_sequences_torch(self):
        seq_array = self.get_sequences_numpy()
        return torch.tensor(seq_array, dtype=torch.float32, device=self.device)
