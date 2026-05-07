"""
data/generate_data.py
---------------------
Synthetic market data generators for the deep hedging project.

Two models are implemented:
  1. HestonSimulator       — Stochastic volatility (Euler–Maruyama)
  2. MertonJumpSimulator   — Jump-diffusion (adds Poisson jumps on top of GBM)

Both return (S_paths, v_paths) of shape (n_paths, T_DAYS+1) suitable for
direct consumption by HedgingEnv.
"""

import numpy as np
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import config


class HestonSimulator:
    """
    Heston (1993) stochastic volatility model.

    dS  = mu * S * dt  + sqrt(v) * S * dW1
    dv  = kappa*(theta - v)*dt + sigma_v*sqrt(v)*dW2
    corr(dW1, dW2) = rho

    The log-Euler scheme is used for S to keep prices strictly positive.
    Variance is reflected at zero to avoid negative values.
    """

    def __init__(
        self,
        S0=config.S0,
        v0=config.V0,
        mu=config.MU,
        kappa=config.KAPPA,
        theta=config.THETA,
        sigma_v=config.SIGMA_V,
        rho=config.RHO,
        T_days=config.T_DAYS,
        dt=config.DT,
    ):
        self.S0 = S0
        self.v0 = v0
        self.mu = mu
        self.kappa = kappa
        self.theta = theta
        self.sigma_v = sigma_v
        self.rho = rho
        self.T_days = T_days
        self.dt = dt

    def simulate(self, n_paths: int, seed: int = None) -> tuple:
        """
        Simulate n_paths independent Heston paths.

        Returns
        -------
        S : np.ndarray, shape (n_paths, T_days+1)
        v : np.ndarray, shape (n_paths, T_days+1)
        """
        if seed is not None:
            np.random.seed(seed)

        n_steps = self.T_days
        S = np.zeros((n_paths, n_steps + 1), dtype=np.float32)
        v = np.zeros((n_paths, n_steps + 1), dtype=np.float32)
        S[:, 0] = self.S0
        v[:, 0] = self.v0

        sqrt_dt = np.sqrt(self.dt)
        rho2 = np.sqrt(max(1.0 - self.rho ** 2, 1e-10))

        for t in range(n_steps):
            v_pos = np.maximum(v[:, t], 0.0)
            sigma_sqrt = np.sqrt(v_pos)

            Z1 = np.random.standard_normal(n_paths).astype(np.float32)
            Z2 = (self.rho * Z1 + rho2 * np.random.standard_normal(n_paths)).astype(np.float32)

            # Log-Euler for S (exact for constant vol)
            S[:, t + 1] = S[:, t] * np.exp(
                (self.mu - 0.5 * v_pos) * self.dt + sigma_sqrt * sqrt_dt * Z1
            )

            # Euler-Maruyama for v with zero-reflection
            v[:, t + 1] = np.maximum(
                v[:, t]
                + self.kappa * (self.theta - v_pos) * self.dt
                + self.sigma_v * sigma_sqrt * sqrt_dt * Z2,
                0.0,
            )

        return S, v


class MertonJumpSimulator:
    """
    Merton (1976) Jump-Diffusion model layered on top of Heston.

    Jump component:  Poisson(lambda*dt) jumps per step,
                     each jump size ~ LogNormal(mu_J, sigma_J^2).
    Variance process follows Heston (no jump in variance here).
    """

    def __init__(
        self,
        S0=config.S0,
        v0=config.V0,
        mu=config.MU,
        kappa=config.KAPPA,
        theta=config.THETA,
        sigma_v=config.SIGMA_V,
        rho=config.RHO,
        lam=config.JUMP_INTENSITY,
        mu_J=config.JUMP_MEAN,
        sigma_J=config.JUMP_STD,
        T_days=config.T_DAYS,
        dt=config.DT,
    ):
        self.heston = HestonSimulator(S0, v0, mu, kappa, theta, sigma_v, rho, T_days, dt)
        self.lam = lam
        self.mu_J = mu_J
        self.sigma_J = sigma_J
        self.dt = dt

    def simulate(self, n_paths: int, seed: int = None) -> tuple:
        """
        Simulate Heston paths and add Merton jumps.

        Returns
        -------
        S : np.ndarray, shape (n_paths, T_days+1)
        v : np.ndarray, shape (n_paths, T_days+1)
        """
        if seed is not None:
            np.random.seed(seed)

        S, v = self.heston.simulate(n_paths)
        n_steps = S.shape[1] - 1

        # Jump compensation drift
        k = np.exp(self.mu_J + 0.5 * self.sigma_J ** 2) - 1.0

        for t in range(n_steps):
            # Number of jumps in [t, t+dt]
            n_jumps = np.random.poisson(self.lam * self.dt, size=n_paths)
            # Aggregate jump size (sum of log-normals)
            jump_sizes = np.ones(n_paths, dtype=np.float32)
            for i in range(n_paths):
                if n_jumps[i] > 0:
                    log_jumps = (
                        self.mu_J + self.sigma_J * np.random.standard_normal(n_jumps[i])
                    )
                    jump_sizes[i] = np.exp(np.sum(log_jumps))
            # Apply jump to the next price
            S[:, t + 1] *= jump_sizes * np.exp(-self.lam * k * self.dt)

        return S, v


def generate_and_cache(model: str = "heston", n_paths: int = config.N_TRAIN_PATHS, seed: int = 42):
    """
    Generate and optionally cache paths to disk.

    Parameters
    ----------
    model   : 'heston' or 'merton'
    n_paths : number of paths to generate
    seed    : random seed for reproducibility
    """
    os.makedirs(config.DATA_CACHE, exist_ok=True)
    cache_S = os.path.join(config.DATA_CACHE, f"{model}_{n_paths}_S.npy")
    cache_v = os.path.join(config.DATA_CACHE, f"{model}_{n_paths}_v.npy")

    if os.path.exists(cache_S) and os.path.exists(cache_v):
        print(f"[data] Loading cached {model} paths from {config.DATA_CACHE}...")
        S = np.load(cache_S)
        v = np.load(cache_v)
        return S, v

    print(f"[data] Generating {n_paths:,} {model} paths (seed={seed})...")
    if model == "heston":
        sim = HestonSimulator()
    elif model == "merton":
        sim = MertonJumpSimulator()
    else:
        raise ValueError(f"Unknown model: {model}")

    S, v = sim.simulate(n_paths, seed=seed)
    np.save(cache_S, S)
    np.save(cache_v, v)
    print(f"[data] Saved to {config.DATA_CACHE}. Shape: S={S.shape}, v={v.shape}")
    return S, v


if __name__ == "__main__":
    # Quick sanity check
    sim = HestonSimulator()
    S, v = sim.simulate(n_paths=1000, seed=0)
    print(f"Heston — S: mean={S[:,-1].mean():.2f}, std={S[:,-1].std():.2f}")
    print(f"Heston — annualised vol: {np.sqrt(v[:,-1].mean()) * 100:.1f}%")

    sim2 = MertonJumpSimulator()
    S2, v2 = sim2.simulate(n_paths=500, seed=1)
    print(f"Merton — S: mean={S2[:,-1].mean():.2f}, std={S2[:,-1].std():.2f}")
