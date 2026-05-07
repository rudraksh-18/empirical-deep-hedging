"""
config.py — Central configuration for Empirical Deep Hedging project.
All hyperparameters are defined here for easy tuning.
"""

# ── Market / Option Parameters ──────────────────────────────────────────────
S0          = 100.0      # Initial stock price
K           = 100.0      # Strike price (ATM)
T_DAYS      = 63         # Option tenor in trading days (~3 months)
DT          = 1 / 252    # Daily time step (1 trading day)
RISK_FREE   = 0.05       # Risk-free rate (annualised)

# ── Heston Stochastic Volatility Model Parameters (calibrated to SPX) ───────
MU          = 0.05       # Drift
KAPPA       = 2.0        # Mean-reversion speed of variance
THETA       = 0.04       # Long-run variance  (~20% annual vol)
SIGMA_V     = 0.3        # Volatility-of-volatility
RHO         = -0.7       # Correlation (leverage effect)
V0          = 0.04       # Initial variance (~20% annual vol)

# ── Merton Jump-Diffusion Parameters ────────────────────────────────────────
JUMP_INTENSITY  = 1.0    # Expected jumps per year (lambda)
JUMP_MEAN       = -0.05  # Mean log jump size (mu_J)
JUMP_STD        = 0.10   # Std of log jump size (sigma_J)

# ── Transaction Costs ────────────────────────────────────────────────────────
TC          = 0.001      # Proportional transaction cost (10 bps)

# ── Environment / Observation Space ─────────────────────────────────────────
OBS_DIM     = 5          # Features per time-step: [log_moneyness, tau, sigma, delta, log_ret]
SEQ_LEN     = 20         # LSTM history window (timesteps)

# ── Data Generation ──────────────────────────────────────────────────────────
N_TRAIN_PATHS   = 50_000   # Pre-generated training paths
N_EVAL_PATHS    = 5_000    # Evaluation paths

# ── Neural Network Architecture ──────────────────────────────────────────────
LSTM_HIDDEN     = 128
LSTM_LAYERS     = 2
FC_HIDDEN       = 256

# ── TD3 / DDPG Hyperparameters ───────────────────────────────────────────────
LR_ACTOR        = 3e-4
LR_CRITIC       = 3e-4
GAMMA           = 0.999      # High discount — hedging is long-horizon episodic
TAU             = 0.005      # Soft-update coefficient for target networks
POLICY_NOISE    = 0.10       # Noise added to target actions (TD3 smoothing)
NOISE_CLIP      = 0.25       # Clipping range for target policy noise
POLICY_DELAY    = 2          # Actor updated every N critic updates
BATCH_SIZE      = 256
BUFFER_SIZE     = 200_000
WARMUP_EPISODES = 100        # Random exploration before learning starts
TOTAL_EPISODES  = 5_000      # Total training episodes
EVAL_EVERY      = 250        # Evaluate every N episodes
SAVE_EVERY      = 500        # Checkpoint every N episodes

# ── Exploration ───────────────────────────────────────────────────────────────
EXPLORE_NOISE   = 0.10       # Gaussian noise std during training

# ── Risk Measure ─────────────────────────────────────────────────────────────
RISK_LAMBDA     = 1.0        # Weight on CVaR in combined loss (mean - lambda*CVaR)
CVAR_ALPHA      = 0.95       # CVaR quantile level

# ── Paths ─────────────────────────────────────────────────────────────────────
CHECKPOINT_DIR  = "checkpoints"
RESULTS_DIR     = "results"
DATA_CACHE      = "data/cache"
