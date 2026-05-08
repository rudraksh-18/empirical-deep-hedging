"""
envs/hedging_env.py
Reward:
  r_t = (delta_t * ΔS_t - tc * |delta_t - delta_{t-1}| * S_t) / option_premium
  r_T += -max(S_T - K, 0) / option_premium
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import config
from benchmarks.black_scholes import bs_call_price


class HedgingEnv(gym.Env):
    """Gymnasium environment for European call option delta-hedging."""

    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        paths_S: np.ndarray = None,
        paths_v: np.ndarray = None,
        tc: float = config.TC,
        seq_len: int = config.SEQ_LEN,
        mode: str = "train",          # 'train' or 'eval'
    ):
        super().__init__()

        self.tc = tc
        self.seq_len = seq_len
        self.obs_dim = config.OBS_DIM
        self.T = config.T_DAYS
        self.K = config.K
        self.S0 = config.S0
        self.dt = config.DT
        self.mode = mode

        # Pre-generated paths (shape: n_paths x T+1)
        self.paths_S = paths_S
        self.paths_v = paths_v
        self.n_paths = len(paths_S) if paths_S is not None else 0

        # Normalisation constant — initial ATM option price under BS
        self.option_premium = bs_call_price(
            self.S0, self.K, self.T * self.dt, config.RISK_FREE, np.sqrt(config.V0)
        )
        self.option_premium = max(self.option_premium, 1e-6)  # guard against zero

        # Gymnasium spaces
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(1,), dtype=np.float32
        )
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(self.seq_len, self.obs_dim),
            dtype=np.float32,
        )

        # Episode state (initialised in reset)
        self._reset_state()



    def _reset_state(self):
        self.step_idx = 0
        self.current_delta = 0.0
        self.total_pnl = 0.0
        self.path_S: np.ndarray = None
        self.path_v: np.ndarray = None
        self.obs_buffer = np.zeros((self.seq_len, self.obs_dim), dtype=np.float32)

    def _get_raw_obs(self, t: int) -> np.ndarray:

        S_t = self.path_S[t]
        v_t = self.path_v[t]

        log_moneyness = np.log(S_t / self.K)
        tau = (self.T - t) * self.dt          # time to expiry (years)
        sigma = np.sqrt(max(v_t, 1e-8))       # instantaneous vol
        delta = self.current_delta
        if t > 0:
            log_ret = np.log(S_t / self.path_S[t - 1])
        else:
            log_ret = 0.0

        obs = np.array(
            [log_moneyness, tau, sigma, delta, log_ret], dtype=np.float32
        )
        return np.clip(obs, -10.0, 10.0)

    def _push_obs(self, raw_obs: np.ndarray):
        """Shift the rolling buffer and append the latest observation."""
        self.obs_buffer = np.roll(self.obs_buffer, shift=-1, axis=0)
        self.obs_buffer[-1] = raw_obs

    def _sample_path(self):
        """Draw a random episode path."""
        if self.paths_S is not None and self.n_paths > 0:
            idx = np.random.randint(0, self.n_paths)
            self.path_S = self.paths_S[idx]
            self.path_v = self.paths_v[idx]
        else:
            from data.generate_data import HestonSimulator
            sim = HestonSimulator()
            S, v = sim.simulate(n_paths=1)
            self.path_S = S[0]
            self.path_v = v[0]



    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self._reset_state()
        self._sample_path()

        raw = self._get_raw_obs(0)
        self._push_obs(raw)

        return self.obs_buffer.copy(), {}

    def step(self, action):
        t = self.step_idx
        action = np.atleast_1d(action)
        new_delta = float(np.clip(action[0], -1.0, 1.0))

        S_t = self.path_S[t]
        S_next = self.path_S[t + 1]

        # P&L from holding new_delta units through [t, t+1]
        hedging_gain = new_delta * (S_next - S_t)

        # Transaction cost for rebalancing
        tc_cost = self.tc * abs(new_delta - self.current_delta) * S_t

        step_pnl = hedging_gain - tc_cost
        self.total_pnl += step_pnl

        # Normalised step reward
        reward = step_pnl / self.option_premium

        # Advance state
        self.current_delta = new_delta
        self.step_idx += 1

        done = self.step_idx >= self.T
        if done:
            # Option payoff at maturity (short call)
            payoff = max(S_next - self.K, 0.0)
            terminal_pnl = -payoff
            self.total_pnl += terminal_pnl
            reward += terminal_pnl / self.option_premium

        # Update observation
        raw = self._get_raw_obs(self.step_idx if not done else t + 1)
        self._push_obs(raw)

        info = {"pnl": self.total_pnl, "step": self.step_idx}
        return self.obs_buffer.copy(), float(reward), done, False, info

    def render(self):
        t = self.step_idx
        S = self.path_S[t] if t < len(self.path_S) else self.path_S[-1]
        print(
            f"Step {t:>3d}/{self.T} | S={S:.2f} | delta={self.current_delta:.3f} | pnl={self.total_pnl:.4f}"
        )
