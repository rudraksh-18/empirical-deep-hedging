"""
agents/ddpg_agent.py
---------------------
Deep Deterministic Policy Gradient (DDPG) agent — baseline comparison.

DDPG is the *original* algorithm used in early deep hedging studies.
It suffers from Q over-estimation and training instability, which is why
the paper (and our project) uses TD3 instead.  We include DDPG to reproduce
the paper's finding that TD3 is significantly more stable and accurate.

Differences from TD3:
  - Single critic (no clipped double-Q)
  - No delayed policy updates
  - No target policy smoothing
"""

import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import config
from models.actor  import LSTMActor
from models.critic import TwinCritic   # reuse — we'll only call .Q1
from agents.replay_buffer import ReplayBuffer


class DDPGAgent:
    """DDPG agent (single critic, no delayed updates, no smoothing)."""

    def __init__(self, device: str = "cpu"):
        self.device = torch.device(device)

        self.actor  = LSTMActor().to(self.device)
        self.critic = TwinCritic().to(self.device)   # only Q1 used

        self.actor_target  = copy.deepcopy(self.actor)
        self.critic_target = copy.deepcopy(self.critic)
        self.actor_target.eval()
        self.critic_target.eval()

        for p in self.actor_target.parameters():  p.requires_grad = False
        for p in self.critic_target.parameters(): p.requires_grad = False

        self.actor_opt  = torch.optim.Adam(self.actor.parameters(),  lr=config.LR_ACTOR)
        self.critic_opt = torch.optim.Adam(self.critic.parameters(), lr=config.LR_CRITIC)

        self.buffer = ReplayBuffer(device=self.device)
        self.total_it = 0
        self.actor_losses:  list = []
        self.critic_losses: list = []

    @torch.no_grad()
    def select_action(self, obs_seq: np.ndarray, add_noise: bool = True) -> np.ndarray:
        obs_t  = torch.FloatTensor(obs_seq).unsqueeze(0).to(self.device)
        action = self.actor(obs_t).squeeze(0).cpu().numpy()
        if add_noise:
            noise  = np.random.normal(0, config.EXPLORE_NOISE, size=action.shape)
            action = np.clip(action + noise, -1.0, 1.0)
        return action.astype(np.float32)

    def train_step(self, batch_size: int = config.BATCH_SIZE) -> dict:
        if len(self.buffer) < batch_size:
            return {}

        self.total_it += 1
        batch = self.buffer.sample(batch_size)

        obs      = batch["obs"]
        next_obs = batch["next_obs"]
        actions  = batch["actions"]
        rewards  = batch["rewards"]
        dones    = batch["dones"]

        # ── Single critic update (no clipping, no noise) ───────────────────
        with torch.no_grad():
            next_action = self.actor_target(next_obs).clamp(-1.0, 1.0)
            q_target    = rewards + config.GAMMA * (1.0 - dones) * \
                          self.critic_target.Q1(next_obs, next_action)

        q1 = self.critic.Q1(obs, actions)
        critic_loss = F.mse_loss(q1, q_target)

        self.critic_opt.zero_grad()
        critic_loss.backward()
        nn.utils.clip_grad_norm_(self.critic.parameters(), max_norm=1.0)
        self.critic_opt.step()

        # ── Actor update every step (no delay) ────────────────────────────
        actor_loss = -self.critic.Q1(obs, self.actor(obs)).mean()

        self.actor_opt.zero_grad()
        actor_loss.backward()
        nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=1.0)
        self.actor_opt.step()

        # Soft update
        self._soft_update(self.actor,  self.actor_target)
        self._soft_update(self.critic, self.critic_target)

        self.actor_losses.append(actor_loss.item())
        self.critic_losses.append(critic_loss.item())

        return {
            "actor_loss":  actor_loss.item(),
            "critic_loss": critic_loss.item(),
        }

    def _soft_update(self, src, tgt):
        tau = config.TAU
        for p_src, p_tgt in zip(src.parameters(), tgt.parameters()):
            p_tgt.data.copy_(tau * p_src.data + (1.0 - tau) * p_tgt.data)

    def store(self, obs, action, reward, next_obs, done):
        self.buffer.add(obs, action, reward, next_obs, done)

    def save(self, directory: str, tag: str = "ddpg"):
        os.makedirs(directory, exist_ok=True)
        torch.save(self.actor.state_dict(),  os.path.join(directory, f"{tag}_actor.pt"))
        torch.save(self.critic.state_dict(), os.path.join(directory, f"{tag}_critic.pt"))

    def load(self, directory: str, tag: str = "ddpg"):
        self.actor.load_state_dict(
            torch.load(os.path.join(directory, f"{tag}_actor.pt"),
                       map_location=self.device, weights_only=True))
        self.critic.load_state_dict(
            torch.load(os.path.join(directory, f"{tag}_critic.pt"),
                       map_location=self.device, weights_only=True))
        self.actor_target  = copy.deepcopy(self.actor)
        self.critic_target = copy.deepcopy(self.critic)

    @property
    def name(self) -> str:
        return "DDPG"
