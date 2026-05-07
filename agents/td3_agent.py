"""
agents/td3_agent.py
--------------------
Twin Delayed Deep Deterministic Policy Gradient (TD3) agent.

Key mechanisms (improvements over DDPG):
  1. Clipped Double Q-Learning  — two critics; use min(Q1, Q2) as target → 
                                   prevents Q over-estimation
  2. Delayed Policy Updates     — actor updated every POLICY_DELAY critic steps →
                                   more stable gradients
  3. Target Policy Smoothing    — add clipped noise to target actions →
                                   prevents exploitation of critic peaks

Reference: Fujimoto et al. (2018) "Addressing Function Approximation Error
in Actor-Critic Methods"  https://arxiv.org/abs/1802.09477
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
from models.critic import TwinCritic
from agents.replay_buffer import ReplayBuffer


class TD3Agent:
    """
    TD3 agent for continuous-action option hedging.

    Parameters
    ----------
    device        : 'cpu' or 'cuda'
    tc            : transaction cost (for env, not agent-internal)
    actor_kwargs  : override LSTMActor defaults
    critic_kwargs : override TwinCritic defaults
    """

    def __init__(self, device: str = "cpu", **kwargs):
        self.device = torch.device(device)

        # Networks
        self.actor  = LSTMActor().to(self.device)
        self.critic = TwinCritic().to(self.device)

        # Target networks (hard copy)
        self.actor_target  = copy.deepcopy(self.actor)
        self.critic_target = copy.deepcopy(self.critic)
        self.actor_target.eval()
        self.critic_target.eval()

        # Freeze target parameters (updated only via soft update)
        for p in self.actor_target.parameters():  p.requires_grad = False
        for p in self.critic_target.parameters(): p.requires_grad = False

        # Optimisers
        self.actor_opt  = torch.optim.Adam(self.actor.parameters(),  lr=config.LR_ACTOR)
        self.critic_opt = torch.optim.Adam(self.critic.parameters(), lr=config.LR_CRITIC)

        # Replay buffer
        self.buffer = ReplayBuffer(device=self.device)

        # Training counters
        self.total_it = 0          # total critic update steps
        self.actor_losses: list = []
        self.critic_losses: list = []

    # ── Action selection ────────────────────────────────────────────────────

    @torch.no_grad()
    def select_action(self, obs_seq: np.ndarray, add_noise: bool = True) -> np.ndarray:
        """
        Given a state sequence, return an action.

        Parameters
        ----------
        obs_seq   : (seq_len, obs_dim) numpy array
        add_noise : add Gaussian exploration noise during training
        """
        obs_t = torch.FloatTensor(obs_seq).unsqueeze(0).to(self.device)  # (1, seq, obs)
        action = self.actor(obs_t).squeeze(0).cpu().numpy()               # (1,)
        if add_noise:
            noise = np.random.normal(0, config.EXPLORE_NOISE, size=action.shape)
            action = np.clip(action + noise, -1.0, 1.0)
        return action.astype(np.float32)

    # ── Learning step ───────────────────────────────────────────────────────

    def train_step(self, batch_size: int = config.BATCH_SIZE) -> dict:
        """
        One TD3 update step.

        Returns dict with scalar loss values for logging.
        """
        if len(self.buffer) < batch_size:
            return {}

        self.total_it += 1
        batch = self.buffer.sample(batch_size)

        obs      = batch["obs"]        # (B, seq, obs_dim)
        next_obs = batch["next_obs"]
        actions  = batch["actions"]    # (B, 1)
        rewards  = batch["rewards"]    # (B, 1)
        dones    = batch["dones"]      # (B, 1)

        # ── Critic update ──────────────────────────────────────────────────
        with torch.no_grad():
            # Target action with smoothing noise
            noise = (
                torch.randn_like(actions) * config.POLICY_NOISE
            ).clamp(-config.NOISE_CLIP, config.NOISE_CLIP)
            next_action = (self.actor_target(next_obs) + noise).clamp(-1.0, 1.0)

            # Clipped double-Q target
            q1_t, q2_t = self.critic_target(next_obs, next_action)
            q_target = rewards + config.GAMMA * (1.0 - dones) * torch.min(q1_t, q2_t)

        q1, q2 = self.critic(obs, actions)
        critic_loss = F.mse_loss(q1, q_target) + F.mse_loss(q2, q_target)

        self.critic_opt.zero_grad()
        critic_loss.backward()
        nn.utils.clip_grad_norm_(self.critic.parameters(), max_norm=1.0)
        self.critic_opt.step()

        logs = {"critic_loss": critic_loss.item()}

        # ── Delayed actor update ───────────────────────────────────────────
        if self.total_it % config.POLICY_DELAY == 0:
            actor_loss = -self.critic.Q1(obs, self.actor(obs)).mean()

            self.actor_opt.zero_grad()
            actor_loss.backward()
            nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=1.0)
            self.actor_opt.step()

            # Soft update targets
            self._soft_update(self.actor,  self.actor_target)
            self._soft_update(self.critic, self.critic_target)

            logs["actor_loss"] = actor_loss.item()
            self.actor_losses.append(actor_loss.item())

        self.critic_losses.append(critic_loss.item())
        return logs

    # ── Utilities ───────────────────────────────────────────────────────────

    def _soft_update(self, src: nn.Module, tgt: nn.Module):
        """θ_target ← τ·θ + (1-τ)·θ_target"""
        tau = config.TAU
        for p_src, p_tgt in zip(src.parameters(), tgt.parameters()):
            p_tgt.data.copy_(tau * p_src.data + (1.0 - tau) * p_tgt.data)

    def store(self, obs, action, reward, next_obs, done):
        """Add a transition to the replay buffer."""
        self.buffer.add(obs, action, reward, next_obs, done)

    def save(self, directory: str, tag: str = "td3"):
        os.makedirs(directory, exist_ok=True)
        torch.save(self.actor.state_dict(),  os.path.join(directory, f"{tag}_actor.pt"))
        torch.save(self.critic.state_dict(), os.path.join(directory, f"{tag}_critic.pt"))
        print(f"[TD3] Saved to {directory}/{tag}_*.pt")

    def load(self, directory: str, tag: str = "td3"):
        self.actor.load_state_dict(
            torch.load(os.path.join(directory, f"{tag}_actor.pt"),
                       map_location=self.device, weights_only=True))
        self.critic.load_state_dict(
            torch.load(os.path.join(directory, f"{tag}_critic.pt"),
                       map_location=self.device, weights_only=True))
        self.actor_target  = copy.deepcopy(self.actor)
        self.critic_target = copy.deepcopy(self.critic)
        print(f"[TD3] Loaded from {directory}/{tag}_*.pt")

    @property
    def name(self) -> str:
        return "TD3"
