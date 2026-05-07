# 📈 Empirical Deep Hedging

> *A Python implementation of Mikkilä & Kanniainen (2023) "Empirical Deep Hedging", with improvements including LSTM actors, CVaR risk measures, and a full TD3 vs DDPG vs Black-Scholes comparison.*

---

## 📖 What Is This Research About?

### The Problem with Classical Hedging

When a bank sells an options contract to a client, it is exposed to risk — if the stock price moves unfavourably, the bank loses money. To manage this, traders **hedge** the option by continuously trading in the underlying stock. The classic approach is **Black-Scholes delta hedging**, which tells you exactly how many shares to hold at each moment.

**But Black-Scholes has serious flaws:**
- It assumes **constant volatility** — in reality, volatility clusters and changes over time
- It assumes **continuous trading** — real markets are discrete
- It assumes **zero transaction costs** — real trading costs money
- It requires **knowing the model** — the true market dynamics are unknown

### The Paper's Solution: Empirical Deep Hedging

**"Empirical Deep Hedging"** by Mikkilä & Kanniainen (2023, *Quantitative Finance*) proposes a radically different approach:

> *Instead of assuming a financial model, train a Deep Reinforcement Learning agent directly on real market data to discover the optimal hedging strategy.*

**Key contributions of the paper:**

| Contribution | Details |
|---|---|
| **Data-Driven** | Agent trained on real S&P 500 intra-day data — no model assumptions |
| **Algorithm** | Uses **TD3** (Twin Delayed DDPG), which is more stable than standard DDPG |
| **Performance** | Outperforms Black-Scholes delta hedge, especially under transaction costs |
| **Model-Free** | Agent learns vol dynamics implicitly from data — no need to "specify" a volatility model |

### What is Deep Reinforcement Learning (DRL)?

Think of the hedging agent as a **player learning to play a game**:
- **State**: Current market conditions (stock price, volatility, time to expiry, current hedge position)
- **Action**: How many shares of stock to hold (the "delta")
- **Reward**: Profit & Loss of the hedged position, minus transaction costs
- **Goal**: Maximise cumulative reward (= minimise hedging risk) over the life of the option

The agent uses a neural network to decide the optimal action at each state, and improves by interacting with the market (or a market simulator) millions of times.

### What is TD3 vs DDPG?

| Algorithm | Description | Issue |
|---|---|---|
| **DDPG** | Original actor-critic for continuous actions | Q-value over-estimation → unstable training |
| **TD3** | Twin Delayed DDPG | Fixes DDPG with 3 improvements (see below) → stable, accurate |

**TD3's three fixes:**
1. **Clipped Double Q-Learning**: Use two critics; take the *minimum* Q-value → prevents over-estimation
2. **Delayed Policy Updates**: Update the actor (strategy) less frequently than the critic → more stable
3. **Target Policy Smoothing**: Add noise to target actions → prevents overfitting to sharp Q peaks

---

## 🏗️ Project Architecture

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
│   ├── actor.py              ← LSTM + MLP actor network (OUR IMPROVEMENT)
│   └── critic.py             ← Twin LSTM critic networks
├── agents/
│   ├── __init__.py
│   ├── replay_buffer.py      ← Sequence-aware experience replay
│   ├── td3_agent.py          ← TD3 agent (main algorithm)
│   └── ddpg_agent.py         ← DDPG baseline for comparison
├── benchmarks/
│   ├── __init__.py
│   └── black_scholes.py      ← Classical BS delta hedge
├── checkpoints/              ← 💾 Trained model weights (generated)
├── results/
│   ├── eval_summary.json     ← Evaluation metrics (generated)
│   └── figs/                 ← 📈 Generated charts (generated)
├── train.py                  ← 🚀 Main training script
├── evaluate.py               ← 📊 Evaluation and comparison
├── visualize.py              ← 📈 Publication-quality charts
├── config.py                 ← All hyperparameters
├── requirements.txt
├── setup_env.bat             ← Windows environment setup
└── README.md
```

---

## 🚀 Quick Start

### Step 1: Set up the environment

```bash
# Run the setup script (creates venv, installs all dependencies)
setup_env.bat

# Or manually:
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

### Step 2: Train the agents

```bash
# Train both TD3 and DDPG (full run, ~1-2 hours on CPU)
python train.py

# Train only TD3 with Merton jump-diffusion data
python train.py --agent td3 --model merton

# Customise transaction costs
python train.py --tc 0.005   # 50 bps
```

### Step 3: Evaluate results

```bash
# Compare all strategies
python evaluate.py

# Sweep across transaction cost levels
python evaluate.py --tc 0.0 0.001 0.005 0.01
```

### Step 4: Generate charts

```bash
python visualize.py
# Charts saved to results/figs/
```

---

## 🔬 Our Improvements Over the Paper

| Feature | Paper | Our Implementation |
|---|---|---|
| **Actor Network** | Plain MLP | **LSTM + MLP** (captures path-dependent vol dynamics) |
| **Risk Measure** | Variance | **CVaR-95** (Expected Shortfall — more robust to tail risk) |
| **Training Stability** | Gradient clipping | + **LayerNorm** on LSTM hidden states |
| **Data** | Real SPX (proprietary) | **Heston + Merton** synthetic (calibrated to SPX statistics) |
| **Baselines** | BS delta only | BS delta + **DDPG** (shows why TD3 matters) |
| **Visualisations** | Minimal | **5 chart types** including TC sweep and example paths |

### Why LSTM?

The original paper uses a Multi-Layer Perceptron (MLP) that sees only the *current* state. An LSTM maintains a **hidden memory** of the past 20 trading days, enabling it to:
- Detect volatility clustering (high-vol days tend to cluster)
- Adapt to trending markets vs mean-reverting regimes
- Learn the "momentum" of the hedge ratio over time

```
           MLP Actor (paper)           LSTM Actor (ours)
           ─────────────────           ──────────────────
Input:     [S, τ, σ, δ, r]             [S_t, τ, σ, δ, r] for t-20..t
           (just current state)         (history window)
             ↓ FC layers                 ↓ LSTM → LayerNorm
           action                        ↓ FC layers
                                        action
```

### Why CVaR instead of Variance?

CVaR-95 (Conditional Value at Risk) measures the **average loss in the worst 5% of scenarios**. This is:
- More sensitive to tail risk (rare but catastrophic losses)
- A standard risk metric in financial regulation (Basel III)
- Better aligned with how risk managers actually think

---

## ⚙️ Configuration

All hyperparameters are in `config.py`. Key settings:

```python
# Market model
T_DAYS   = 63       # 3-month option tenor
TC       = 0.001    # 10 bps transaction cost

# Heston model (calibrated to SPX)
KAPPA    = 2.0      # Mean-reversion speed
THETA    = 0.04     # Long-run variance (20% vol)
SIGMA_V  = 0.3      # Vol-of-vol
RHO      = -0.7     # Leverage effect (negative correlation)

# TD3 training
TOTAL_EPISODES  = 5000
BATCH_SIZE      = 256
LR_ACTOR        = 3e-4
POLICY_DELAY    = 2   # Actor update frequency
```

---

## 📊 Expected Results

After full training you should see approximately:

| Strategy | Mean P&L | Std | CVaR-95 |
|---|---|---|---|
| Black-Scholes | -0.05 to -0.15 | high | high |
| DDPG | better than BS | medium | medium |
| **TD3** | **best** | **lowest** | **lowest** |

*(Exact numbers depend on TC level and market model — higher TC hurts BS more than RL agents.)*

---

## 📚 References

1. **Mikkilä, O. & Kanniainen, J. (2023).** "Empirical deep hedging." *Quantitative Finance*, 23(1), 111–122.
2. **Bühler, H., Gonon, L., Teichmann, J., & Wood, B. (2019).** "Deep hedging." *Quantitative Finance*, 19(8), 1271–1291.
3. **Fujimoto, S., Hoof, H., & Meger, D. (2018).** "Addressing function approximation error in actor-critic methods (TD3)." *ICML 2018.*
4. **Heston, S. L. (1993).** "A closed-form solution for options with stochastic volatility." *Review of Financial Studies*, 6(2), 327–343.
5. **Merton, R. C. (1976).** "Option pricing when underlying stock returns are discontinuous." *Journal of Financial Economics*, 3(1-2), 125–144.

---

## 🤝 Glossary

| Term | Meaning |
|---|---|
| **Delta** | Sensitivity of option price to the underlying price (∂C/∂S). BS delta is used as the hedge ratio. |
| **Hedging** | Holding an offsetting position in the underlying to neutralise risk from the option |
| **Implied Vol** | The volatility that, when put into the BS formula, reproduces the market price |
| **CVaR-95** | Average loss in the worst 5% of scenarios (tail risk measure) |
| **Actor** | Neural network that maps state → action (the "policy") |
| **Critic** | Neural network that maps (state, action) → Q-value (expected future reward) |
| **Replay Buffer** | Memory bank of past transitions sampled randomly for training |
| **Soft Update** | Slowly blend target network weights: θ_target ← τ·θ + (1-τ)·θ_target |
