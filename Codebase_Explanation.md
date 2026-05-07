# 🧠 Empirical Deep Hedging: Comprehensive Codebase Walkthrough

This document explains every single folder and file in the project, what the code does, and exactly how everything connects when you run the project. You can use this as your "cheat sheet" for your presentation.

---

## 1. How Everything Connects (Execution Flow)
When you run `python train.py`, here is the step-by-step flow of what happens under the hood:

1. **Data Generation:** `train.py` calls `data/generate_data.py` to simulate thousands of hypothetical stock market paths (stock prices and volatility) using the Heston or Merton models.
2. **Environment Setup:** These paths are loaded into `envs/hedging_env.py`. This acts like a "video game" where the AI will practice trading.
3. **Agent Creation:** The `TD3Agent` is created from `agents/td3_agent.py`. It initializes its brain, which consists of the neural networks found in `models/actor.py` and `models/critic.py`.
4. **The Training Loop:**
   - The environment gives the Agent the current market state (stock price, time to expiry, etc.).
   - The Agent's **Actor** network predicts an action: "Hold X amount of shares" (the delta).
   - The Environment steps forward 1 day, calculates the Profit/Loss (P&L) and charges transaction costs. It returns a **Reward** to the agent.
   - The Agent stores this memory in `agents/replay_buffer.py`.
   - The Agent's **Critic** network evaluates how good that action was, and updates the Actor network to be smarter next time.
5. **Benchmarking:** Finally, `evaluate.py` compares the AI's performance against the classical model found in `benchmarks/black_scholes.py`.

---

## 2. Folder-by-Folder Breakdown

### 📁 `data/` (Market Simulators)
This folder is responsible for generating the synthetic stock market data the agent trains on.
*   **`generate_data.py`**: Instead of using limited historical data, this file uses Monte Carlo simulations to generate infinite future stock paths.
    *   *HestonSimulator:* Simulates prices where the volatility is constantly changing (Stochastic Volatility).
    *   *MertonSimulator:* Simulates prices with sudden "jumps" or market crashes (Jump-Diffusion).

### 📁 `envs/` (The Trading Environment)
This is the "gym" where the AI agent trains. It follows the standard OpenAI `Gymnasium` structure.
*   **`hedging_env.py`**:
    *   `reset()`: Starts a new 63-day option contract and picks a random stock path.
    *   `step(action)`: The agent submits its target hedge ratio (action). The environment steps forward one day, calculates the cost of buying/selling the stock (Transaction Costs), and calculates the hedging gain/loss.
    *   *Reward Calculation:* The reward is strictly the P&L minus transaction costs. At the very last step (maturity), it also subtracts the final payout of the options contract.

### 📁 `models/` (The Neural Networks)
This folder holds the raw PyTorch Neural Network architectures.
*   **`actor.py` (The Strategy)**: The network that decides *what to do*. We implemented an **LSTM (Long Short-Term Memory)** network here. It takes a sequence of the last 20 days of market data, processes it through memory cells (to detect trends/volatility clusters), and outputs a single number between -1 and 1 representing the target stock holding (delta).
*   **`critic.py` (The Evaluator)**: The network that decides *how good the action was*. It takes the market state AND the Actor's chosen action, and predicts the Expected Reward (Q-Value). We use a "Twin Critic" (two networks) to prevent the AI from overestimating how much money it will make.

### 📁 `agents/` (The RL Algorithms)
This folder contains the actual Reinforcement Learning logic that trains the neural networks.
*   **`td3_agent.py`**: The core AI of the project (Twin Delayed DDPG). 
    *   `select_action()`: Passes the state to the Actor network to get a trading decision. During training, it adds random Gaussian noise to explore new strategies.
    *   `train_step()`: Samples a batch of past memories. It updates the Critic to accurately guess rewards, and updates the Actor to maximize the Critic's score.
    *   *Key Feature:* "Delayed Updates" - It only updates the Actor network once every two times it updates the Critic, which makes the math much more stable.
*   **`ddpg_agent.py`**: A simpler, older algorithm included purely so we can compare it to TD3 and prove TD3 is better.
*   **`replay_buffer.py`**: A memory bank. As the agent trades, it stores (State, Action, Reward, Next State) tuples here. The agent learns by pulling random batches from this memory.

### 📁 `benchmarks/`
*   **`black_scholes.py`**: The classical 1973 math formula used by Wall Street. It calculates the theoretical option price and the theoretical Delta. We use this as our "enemy" to beat. Because it assumes zero transaction costs, the AI agent can beat it in our realistic environment.

### 📁 Root Level Scripts (The Execution files)
*   **`config.py`**: The control center. Every hyperparameter (learning rates, transaction cost amounts, option strike price, number of training days) is stored here. Changing a number here changes it everywhere in the project.
*   **`train.py`**: The script that orchestrates pulling data, building the environment, and running the agent through thousands of episodes. It saves checkpoints to the `checkpoints/` folder.
*   **`evaluate.py`**: Loads the trained brain from `checkpoints/` and runs it on *brand new, unseen* stock paths to see how it performs in the real world. It compares it to Black-Scholes and saves the metrics to `results/eval_summary.json`.
*   **`visualize.py`**: Reads the `.json` results and generates the beautiful `matplotlib` graphs found in `results/figs/`.
