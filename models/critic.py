"""
models/critic.py
----------------
Twin LSTM + MLP Critic networks for TD3.

Each critic estimates Q(s_seq, a):
  Input : obs_seq (batch, seq_len, obs_dim)  + action (batch, 1)
  Output: scalar Q-value (batch, 1)

Using *two* independent critics (Q1, Q2) is the core of TD3's clipped
double Q-learning trick that prevents over-estimation bias.

Architecture per critic
-----------------------
  obs_seq  → LSTM → LayerNorm → h
  concat(h, action)
  → FC(lstm_hidden+1 → fc_hidden), ReLU
  → FC(fc_hidden → fc_hidden//2), ReLU
  → FC(fc_hidden//2 → 1)
"""

import torch
import torch.nn as nn
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import config


class _SingleCritic(nn.Module):
    """One LSTM-based Q-network."""

    def __init__(self, obs_dim, action_dim, lstm_hidden, lstm_layers, fc_hidden):
        super().__init__()

        self.lstm = nn.LSTM(
            input_size=obs_dim,
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            batch_first=True,
            dropout=0.1 if lstm_layers > 1 else 0.0,
        )
        self.norm = nn.LayerNorm(lstm_hidden)

        self.fc = nn.Sequential(
            nn.Linear(lstm_hidden + action_dim, fc_hidden),
            nn.ReLU(),
            nn.Linear(fc_hidden, fc_hidden // 2),
            nn.ReLU(),
            nn.Linear(fc_hidden // 2, 1),
        )

        self._init_weights()

    def _init_weights(self):
        for name, p in self.lstm.named_parameters():
            if "weight" in name:
                nn.init.orthogonal_(p)
            elif "bias" in name:
                nn.init.zeros_(p)
        for m in self.fc:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, obs_seq: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        self.lstm.flatten_parameters()
        _, (h_n, _) = self.lstm(obs_seq)
        h = self.norm(h_n[-1])                   # (batch, lstm_hidden)
        x = torch.cat([h, action], dim=-1)        # (batch, lstm_hidden + action_dim)
        return self.fc(x)                         # (batch, 1)


class TwinCritic(nn.Module):
    """
    Two independent critics (Q1, Q2) as required by TD3.

    Usage
    -----
    q1, q2 = critic(obs_seq, action)   # both (batch, 1)
    q1     = critic.Q1(obs_seq, action)
    """

    def __init__(
        self,
        obs_dim:     int = config.OBS_DIM,
        action_dim:  int = 1,
        lstm_hidden: int = config.LSTM_HIDDEN,
        lstm_layers: int = config.LSTM_LAYERS,
        fc_hidden:   int = config.FC_HIDDEN,
    ):
        super().__init__()
        args = (obs_dim, action_dim, lstm_hidden, lstm_layers, fc_hidden)
        self.q1_net = _SingleCritic(*args)
        self.q2_net = _SingleCritic(*args)

    def forward(self, obs_seq, action):
        return self.q1_net(obs_seq, action), self.q2_net(obs_seq, action)

    def Q1(self, obs_seq, action):
        return self.q1_net(obs_seq, action)


if __name__ == "__main__":
    batch, seq_len, obs_dim = 32, config.SEQ_LEN, config.OBS_DIM
    critic = TwinCritic()
    obs = torch.randn(batch, seq_len, obs_dim)
    act = torch.randn(batch, 1)
    q1, q2 = critic(obs, act)
    print(f"Q1 shape: {q1.shape}, Q2 shape: {q2.shape}")
    n_params = sum(p.numel() for p in critic.parameters() if p.requires_grad)
    print(f"Trainable parameters: {n_params:,}")
