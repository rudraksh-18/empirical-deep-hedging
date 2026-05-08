"""
models/actor.py
"""

import torch
import torch.nn as nn
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import config


class LSTMActor(nn.Module):
    """LSTM + MLP policy network. Outputs a scalar action ∈ (-1, 1)."""

    def __init__(
        self,
        obs_dim:     int = config.OBS_DIM,
        lstm_hidden: int = config.LSTM_HIDDEN,
        lstm_layers: int = config.LSTM_LAYERS,
        fc_hidden:   int = config.FC_HIDDEN,
    ):
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
            nn.Linear(lstm_hidden, fc_hidden),
            nn.ReLU(),
            nn.Linear(fc_hidden, fc_hidden // 2),
            nn.ReLU(),
            nn.Linear(fc_hidden // 2, 1),
            nn.Tanh(),
        )

        self._init_weights()

    def _init_weights(self):
        """Orthogonal initialisation for LSTM."""
        for name, p in self.lstm.named_parameters():
            if "weight" in name:
                nn.init.orthogonal_(p)
            elif "bias" in name:
                nn.init.zeros_(p)
        for m in self.fc:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)
        # Small output layer — keeps initial actions near zero
        nn.init.uniform_(self.fc[-2].weight, -3e-3, 3e-3)
        nn.init.zeros_(self.fc[-2].bias)

    def forward(self, obs_seq: torch.Tensor) -> torch.Tensor:

        self.lstm.flatten_parameters()
        # LSTM — only keep last hidden state
        _, (h_n, _) = self.lstm(obs_seq)
        # h_n: (num_layers, batch, lstm_hidden) → take top layer
        h = h_n[-1]               # (batch, lstm_hidden)
        h = self.norm(h)
        return self.fc(h)         # (batch, 1)


if __name__ == "__main__":
    batch, seq_len, obs_dim = 32, config.SEQ_LEN, config.OBS_DIM
    actor = LSTMActor()
    x = torch.randn(batch, seq_len, obs_dim)
    y = actor(x)
    print(f"Actor output shape: {y.shape}  | range: [{y.min():.3f}, {y.max():.3f}]")
    n_params = sum(p.numel() for p in actor.parameters() if p.requires_grad)
    print(f"Trainable parameters: {n_params:,}")
