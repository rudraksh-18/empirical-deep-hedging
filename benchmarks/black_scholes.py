"""
benchmarks/black_scholes.py
----------------------------
Classic Black-Scholes pricing and delta-hedging benchmark.

Functions
---------
bs_call_price   — Closed-form European call price (Black-Scholes)
bs_delta        — BS delta (∂C/∂S)
bs_implied_vol  — Implied volatility via Brent root-finding
run_bs_hedger   — Simulate a full delta-hedging episode; returns episode P&L

The BlackScholesHedger class wraps run_bs_hedger for batch evaluation and
computing summary statistics comparable to the RL agents.
"""

import numpy as np
from scipy.stats import norm
from scipy.optimize import brentq
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import config


# ── Core BS Formulae ─────────────────────────────────────────────────────────

def _d1(S, K, tau, r, sigma):
    """BS d1 term. tau in years."""
    return (np.log(S / K) + (r + 0.5 * sigma ** 2) * tau) / (sigma * np.sqrt(tau))


def bs_call_price(S, K, tau, r, sigma):
    """
    Black-Scholes European call price.

    Parameters
    ----------
    S     : current underlying price
    K     : strike price
    tau   : time to expiry in years
    r     : risk-free rate (annualised)
    sigma : volatility (annualised)
    """
    if tau <= 0:
        return max(S - K, 0.0)
    d1 = _d1(S, K, tau, r, sigma)
    d2 = d1 - sigma * np.sqrt(tau)
    return S * norm.cdf(d1) - K * np.exp(-r * tau) * norm.cdf(d2)


def bs_delta(S, K, tau, r, sigma):
    """Black-Scholes delta for a European call (∂C/∂S)."""
    if tau <= 0:
        return 1.0 if S > K else 0.0
    d1 = _d1(S, K, tau, r, sigma)
    return float(norm.cdf(d1))


def bs_implied_vol(market_price, S, K, tau, r, tol=1e-6, max_iter=200):
    """
    Compute implied volatility from a market call price via Brent's method.
    Returns np.nan if root-finding fails or the price is below intrinsic value.
    """
    intrinsic = max(S - K * np.exp(-r * tau), 0.0)
    if market_price <= intrinsic:
        return np.nan
    try:
        iv = brentq(
            lambda sigma: bs_call_price(S, K, tau, r, sigma) - market_price,
            1e-4,
            10.0,
            xtol=tol,
            maxiter=max_iter,
        )
        return iv
    except ValueError:
        return np.nan


# ── Episode Hedging Simulator ────────────────────────────────────────────────

def run_bs_hedger(
    path_S: np.ndarray,
    path_v: np.ndarray,
    K: float = config.K,
    r: float = config.RISK_FREE,
    dt: float = config.DT,
    tc: float = config.TC,
) -> float:
    """
    Run one Black-Scholes delta-hedging episode on a pre-generated path.

    At each time-step, the hedge ratio is set to the BS delta computed
    with the *true* instantaneous Heston volatility (sqrt(v_t)).  This is the
    "oracle" BS hedge — the best a model-based approach can do given the
    correct volatility.

    Parameters
    ----------
    path_S : 1-D array, shape (T+1,) — underlying price path
    path_v : 1-D array, shape (T+1,) — variance path
    K      : strike
    r      : risk-free rate
    dt     : step size (years)
    tc     : proportional transaction cost

    Returns
    -------
    total_pnl : float — total P&L of the hedged portfolio (normalised)
    """
    T = len(path_S) - 1
    option_premium = bs_call_price(path_S[0], K, T * dt, r, np.sqrt(path_v[0]))
    option_premium = max(option_premium, 1e-6)

    current_delta = 0.0
    total_pnl = 0.0

    for t in range(T):
        S_t = path_S[t]
        v_t = path_v[t]
        sigma_t = np.sqrt(max(v_t, 1e-8))
        tau = (T - t) * dt

        # BS delta with current vol
        new_delta = bs_delta(S_t, K, tau, r, sigma_t)

        S_next = path_S[t + 1]
        hedging_gain = new_delta * (S_next - S_t)
        tc_cost = tc * abs(new_delta - current_delta) * S_t
        total_pnl += hedging_gain - tc_cost
        current_delta = new_delta

    # Option payoff at maturity
    payoff = max(path_S[-1] - K, 0.0)
    total_pnl -= payoff
    return total_pnl / option_premium


# ── Batch Evaluation Helper ──────────────────────────────────────────────────

class BlackScholesHedger:
    """Wrapper for batch evaluation of the BS delta-hedge strategy."""

    def __init__(self, tc: float = config.TC):
        self.tc = tc

    def evaluate(self, paths_S: np.ndarray, paths_v: np.ndarray) -> dict:
        """
        Evaluate BS delta hedge on a batch of paths.

        Returns
        -------
        dict with keys: pnl_mean, pnl_std, pnl_cvar95, hedging_error
        """
        n = len(paths_S)
        pnls = np.array([
            run_bs_hedger(paths_S[i], paths_v[i], tc=self.tc)
            for i in range(n)
        ], dtype=np.float32)

        sorted_pnls = np.sort(pnls)
        cutoff = int(n * (1 - config.CVAR_ALPHA))
        cvar = -float(sorted_pnls[:max(cutoff, 1)].mean())

        return {
            "pnl_mean":      float(pnls.mean()),
            "pnl_std":       float(pnls.std()),
            "pnl_cvar95":    cvar,
            "hedging_error": float(pnls.std()),   # RMSE proxy
            "all_pnls":      pnls,
        }


if __name__ == "__main__":
    # Sanity check
    price = bs_call_price(100, 100, 0.25, 0.05, 0.20)
    delta = bs_delta(100, 100, 0.25, 0.05, 0.20)
    iv    = bs_implied_vol(price, 100, 100, 0.25, 0.05)
    print(f"BS call price: {price:.4f}")
    print(f"BS delta:      {delta:.4f}")
    print(f"Implied vol:   {iv:.4f}  (should be ~0.20)")
