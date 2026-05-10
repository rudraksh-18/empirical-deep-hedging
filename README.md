# Empirical Deep Hedging

> *A Python implementation of Mikkilä & Kanniainen (2023) "Empirical Deep Hedging", with improvements including LSTM actors, CVaR-95 risk measures, and a full TD3 vs DDPG vs Black-Scholes comparison.*

**Team:** BullishGang — Rudraksh Rajendra Lande (B23176) & Suyash Bilmore (B23201)

---

## Required Libraries / Packages

All dependencies are listed in `requirements.txt`. The project requires **Python 3.10 or higher**.

| Package | Version | Purpose |
|---|---|---|
| `torch` | ≥ 2.0.0 | Deep learning framework for all neural network models (Actor, Critic) |
| `numpy` | ≥ 1.24.0 | Numerical computations, Monte Carlo simulations |
| `scipy` | ≥ 1.10.0 | Statistical functions (CVaR computation, distributions) |
| `matplotlib` | ≥ 3.7.0 | Generating all result charts and figures |
| `seaborn` | ≥ 0.12.0 | Enhanced plot styling for P&L distributions |
| `pandas` | ≥ 2.0.0 | Data handling and results tabulation |
| `gymnasium` | ≥ 0.28.0 | OpenAI Gym-compatible trading environment framework |
| `tqdm` | ≥ 4.65.0 | Training progress bars |
| `tensorboard` | ≥ 2.13.0 | Optional: live training curve visualization |

Install all at once:
```bash
pip install -r requirements.txt
```

---

## Dependencies & Setup Instructions

### System Requirements
- **OS:** Windows 10/11, Linux, or macOS
- **Python:** 3.10 or above — download from [python.org](https://www.python.org/downloads/)
- **Hardware:** CPU is sufficient; a CUDA-capable GPU will speed up training significantly
- **Disk:** ~500 MB free space (for venv, model checkpoints, and generated data)

### Setup — Windows (Recommended)

**Option A: Automated setup script (one click)**
```bat
setup_env.bat
```
This script automatically:
1. Creates a Python virtual environment (`venv/`)
2. Installs all packages from `requirements.txt`
3. Verifies the installation

**Option B: Manual setup**
```bash
# 1. Create a virtual environment
python -m venv venv

# 2. Activate the virtual environment
venv\Scripts\activate          # Windows
# source venv/bin/activate     # Linux / macOS

# 3. Install all required packages
pip install -r requirements.txt

# 4. Verify installation (should print no errors)
python -c "import torch, gymnasium, numpy; print('All dependencies OK')"
```

---

## Steps to Run the Code

Run these steps in order from the project root directory (`QuantProject/`).
Activate the virtual environment first:
```bash
venv\Scripts\activate
```

### Step 1 — Train the Agents
```bash
# Train both TD3 and DDPG agents (recommended, ~1-2 hours on CPU)
python train.py

# Train TD3 only (faster)
python train.py --agent td3

# Train with Merton jump-diffusion market model instead of Heston
python train.py --agent td3 --model merton

# Set a custom transaction cost (default is 0.001 = 10 bps)
python train.py --tc 0.005
```
**Output:** Model weights saved to `checkpoints/td3_final.pth` and `checkpoints/ddpg_final.pth`

### Step 2 — Evaluate Results
```bash
# Compare TD3, DDPG, and Black-Scholes on unseen data
python evaluate.py

# Evaluate across multiple transaction cost levels
python evaluate.py --tc 0.0 0.001 0.005 0.01
```
**Output:** Metrics (Mean P&L, CVaR-95, Sharpe Ratio) saved to `results/eval_summary.json`

### Step 3 — Generate Charts
```bash
# Produce all result visualizations
python visualize.py
```
**Output:** Three charts saved to `results/figs/`:
- `training_curves.png` — TD3 vs DDPG learning convergence
- `pnl_distribution.png` — P&L distribution comparison
- `example_path.png` — Example Heston episode with hedge ratios

---

## Project Architecture

```
QuantProject/
├── data/
│   ├── __init__.py
│   └── generate_data.py      ← Heston + Merton jump-diffusion simulators
├── envs/
│   ├── __init__.py
│   └── hedging_env.py        ← Custom Gymnasium hedging environment
├── models/
│   ├── __init__.py
│   ├── actor.py              ← LSTM + MLP actor network (our improvement)
│   └── critic.py             ← Twin LSTM critic networks
├── agents/
│   ├── __init__.py
│   ├── replay_buffer.py      ← Sequence-aware experience replay
│   ├── td3_agent.py          ← TD3 agent (main algorithm)
│   └── ddpg_agent.py         ← DDPG baseline for comparison
├── benchmarks/
│   ├── __init__.py
│   └── black_scholes.py      ← Classical Black-Scholes delta hedge
├── checkpoints/              ← Trained model weights (auto-generated)
├── results/
│   ├── eval_summary.json     ← Evaluation metrics (auto-generated)
│   └── figs/                 ← Result charts (auto-generated)
├── train.py                  ← Main training script
├── evaluate.py               ← Evaluation and strategy comparison
├── visualize.py              ← Chart generation
├── config.py                 ← All hyperparameters in one place
├── requirements.txt          ← All Python dependencies
├── setup_env.bat             ← Automated Windows setup script
└── README.md
```

---

## What Is This Project About?

### The Problem

When a bank sells an options contract, it is exposed to financial risk. To manage this, traders **delta hedge** the option by continuously adjusting a position in the underlying stock. The standard approach is **Black-Scholes delta hedging**, derived from the PDE:

```
∂V/∂t + ½σ²S²(∂²V/∂S²) + rS(∂V/∂S) − rV = 0
```

**Black-Scholes limitations:**
- Assumes **constant volatility** — real markets exhibit volatility clustering
- Assumes **frictionless, continuous trading** — real markets have transaction costs
- Requires **knowing the true market model** — which is unknown in practice

### Our Solution

We train a **TD3 (Twin Delayed DDPG) Reinforcement Learning agent** with an **LSTM memory network** on synthetic market data (Heston stochastic volatility + Merton jump-diffusion). The agent learns to hedge options purely from market interactions, without any assumption about the underlying dynamics.

**Result:** After 5,000 training episodes, the TD3 agent achieves performance **within 5% of Black-Scholes** while surpassing it on Sharpe Ratio — empirically rediscovering the classical hedge from data alone.

---

## Our Improvements Over the Original Paper

| Feature | Paper (Mikkilä & Kanniainen, 2023) | Our Implementation |
|---|---|---|
| **Actor Network** | Plain MLP | **LSTM + MLP** — captures 20-day market memory |
| **Risk Measure** | Variance | **CVaR-95** — Basel III tail-risk standard |
| **Training Stability** | Gradient clipping only | + **LayerNorm** on LSTM hidden states |
| **Data** | Real SPX (proprietary) | **Heston + Merton** synthetic simulators |
| **Baselines** | Black-Scholes only | Black-Scholes + **DDPG** comparison |

---

## Results (TC = 0.001)

| Strategy | Mean P&L | Std Dev | CVaR-95 | Sharpe Ratio |
|---|---|---|---|---|
| Black-Scholes | -0.917 | 0.202 | 1.360 | -4.538 |
| **TD3 (Ours)** | **-0.963** | **0.241** | **1.512** | **-3.991** |
| DDPG | -7.077 | 9.418 | 28.143 | -0.752 |

TD3 Mean P&L is within **5%** of Black-Scholes and achieves a **better Sharpe Ratio** (-3.991 vs -4.538).

---

## Configuration

All hyperparameters are centralised in `config.py`:

```python
# Market model
T_DAYS   = 63       # 3-month option tenor (trading days)
TC       = 0.001    # Transaction cost (10 basis points)

# Heston model (calibrated to S&P 500)
KAPPA    = 2.0      # Volatility mean-reversion speed
THETA    = 0.04     # Long-run variance (≈ 20% annualised vol)
SIGMA_V  = 0.3      # Volatility-of-volatility
RHO      = -0.7     # Leverage effect (stock-vol correlation)

# TD3 training
TOTAL_EPISODES  = 5000
BATCH_SIZE      = 256
LR_ACTOR        = 3e-4
POLICY_DELAY    = 2     # Update actor every 2 critic steps
```

---

## Team Contributions

| Member | Role | Files Owned |
|---|---|---|
| **Rudraksh Rajendra Lande** (B23176) | Lead RL Engineer & Architect | `actor.py`, `critic.py`, `td3_agent.py`, `replay_buffer.py`, `train.py` |
| **Suyash Bilmore** (B23201) | Data & Environment Engineer | `hedging_env.py`, `generate_data.py`, `evaluate.py`, `visualize.py`, `black_scholes.py` |

---

## Reference

Mikkilä, O. & Kanniainen, J. (2023). "Empirical deep hedging." *Quantitative Finance*, 23(1), 111–122.
https://doi.org/10.1080/14697688.2023.2221281

---

## Glossary

| Term | Meaning |
|---|---|
| **Delta** | Sensitivity of option price to underlying stock price (∂C/∂S) — used as hedge ratio |
| **Hedging** | Holding an offsetting stock position to neutralise risk from the option |
| **CVaR-95** | Average loss in the worst 5% of scenarios (Conditional Value-at-Risk) |
| **Actor** | Neural network mapping state → action (the trading policy) |
| **Critic** | Neural network mapping (state, action) → expected future reward (Q-value) |
| **Replay Buffer** | Memory bank of past experiences sampled randomly during training |
| **Soft Update** | Slowly blending target network weights: θ_target ← τ·θ + (1-τ)·θ_target |
