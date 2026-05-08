"""
agents/replay_buffer.py
"""

import numpy as np
import torch
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import config


class ReplayBuffer:
    """Circular replay buffer for sequence-based observations."""

    def __init__(
        self,
        buffer_size: int = config.BUFFER_SIZE,
        obs_dim:     int = config.OBS_DIM,
        seq_len:     int = config.SEQ_LEN,
        action_dim:  int = 1,
        device:      torch.device = torch.device("cpu"),
    ):
        self.max_size   = buffer_size
        self.obs_dim    = obs_dim
        self.seq_len    = seq_len
        self.action_dim = action_dim
        self.device     = device
        self.ptr        = 0
        self.size       = 0

        # Pre-allocate numpy arrays (faster than list of tuples)
        self.obs      = np.zeros((buffer_size, seq_len, obs_dim),  dtype=np.float32)
        self.next_obs = np.zeros((buffer_size, seq_len, obs_dim),  dtype=np.float32)
        self.actions  = np.zeros((buffer_size, action_dim),        dtype=np.float32)
        self.rewards  = np.zeros((buffer_size, 1),                 dtype=np.float32)
        self.dones    = np.zeros((buffer_size, 1),                 dtype=np.float32)



    def add(
        self,
        obs:      np.ndarray,
        action:   np.ndarray,
        reward:   float,
        next_obs: np.ndarray,
        done:     bool,
    ):
        """Store one transition in the buffer (circular write)."""
        self.obs[self.ptr]      = obs
        self.next_obs[self.ptr] = next_obs
        self.actions[self.ptr]  = action
        self.rewards[self.ptr]  = reward
        self.dones[self.ptr]    = float(done)

        self.ptr  = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample(self, batch_size: int = config.BATCH_SIZE) -> dict:
        """Sample a random mini-batch of transitions."""
        idx = np.random.randint(0, self.size, size=batch_size)
        return {
            "obs":      self._to_tensor(self.obs[idx]),
            "next_obs": self._to_tensor(self.next_obs[idx]),
            "actions":  self._to_tensor(self.actions[idx]),
            "rewards":  self._to_tensor(self.rewards[idx]),
            "dones":    self._to_tensor(self.dones[idx]),
        }

    def __len__(self) -> int:
        return self.size



    def _to_tensor(self, arr: np.ndarray) -> torch.Tensor:
        return torch.FloatTensor(arr).to(self.device)
