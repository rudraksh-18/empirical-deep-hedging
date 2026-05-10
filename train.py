"""
train.py
"""

import argparse
import os
import time
import json
import numpy as np
import torch
from tqdm import tqdm

import config
from data.generate_data  import generate_and_cache
from envs.hedging_env    import HedgingEnv
from agents.td3_agent    import TD3Agent
from agents.ddpg_agent   import DDPGAgent




def parse_args():
    parser = argparse.ArgumentParser(description="Train deep hedging agents")
    parser.add_argument("--agent",    type=str,   default="both",
                        choices=["td3", "ddpg", "both"],
                        help="Which agent(s) to train")
    parser.add_argument("--model",    type=str,   default="heston",
                        choices=["heston", "merton"],
                        help="Market simulation model")
    parser.add_argument("--episodes", type=int,   default=config.TOTAL_EPISODES,
                        help="Number of training episodes")
    parser.add_argument("--tc",       type=float, default=config.TC,
                        help="Transaction cost rate")
    parser.add_argument("--device",   type=str,   
                        default="cuda" if torch.cuda.is_available() else "cpu",
                        help="torch device (cpu / cuda)")
    parser.add_argument("--seed",     type=int,   default=42)
    return parser.parse_args()




def evaluate_agent(agent, eval_paths_S, eval_paths_v, tc, n_episodes=200):
    """Roll out the agent deterministically on n_episodes eval paths."""
    env = HedgingEnv(paths_S=eval_paths_S, paths_v=eval_paths_v, tc=tc, mode="eval")
    pnls = []

    for i in range(n_episodes):
        obs, _ = env.reset()
        done   = False
        while not done:
            action       = agent.select_action(obs, add_noise=False)
            obs, _, done, _, info = env.step(action)
        pnls.append(info["pnl"])

    pnls = np.array(pnls)
    # CVaR at 95% (average of worst 5%)
    sorted_pnls = np.sort(pnls)
    cutoff = max(int(n_episodes * (1 - config.CVAR_ALPHA)), 1)
    cvar = -float(sorted_pnls[:cutoff].mean())

    return {
        "mean":  float(pnls.mean()),
        "std":   float(pnls.std()),
        "cvar95": cvar,
    }




def train_agent(agent, train_env, eval_paths_S, eval_paths_v,
                total_episodes, tc, run_name):
    """Train one agent for total_episodes episodes."""
    os.makedirs(config.CHECKPOINT_DIR, exist_ok=True)
    os.makedirs(config.RESULTS_DIR,    exist_ok=True)

    history = {
        "episode":       [],
        "train_pnl":     [],
        "eval_mean":     [],
        "eval_std":      [],
        "eval_cvar95":   [],
        "actor_loss":    [],
        "critic_loss":   [],
        "wall_time":     [],
    }

    t0 = time.time()

    print(f"\n{'='*60}")
    print(f"  Training {agent.name}  |  {total_episodes} episodes")
    print(f"{'='*60}\n")

    pbar = tqdm(range(1, total_episodes + 1), desc=agent.name, unit="ep")
    for ep in pbar:

        obs, _ = train_env.reset()
        done   = False
        ep_pnl = 0.0

        while not done:
            # Warmup: random actions until buffer has enough data
            if ep <= config.WARMUP_EPISODES:
                action = train_env.action_space.sample()
            else:
                action = agent.select_action(obs, add_noise=True)

            next_obs, reward, done, _, info = train_env.step(action)

            agent.store(obs, action, float(reward), next_obs, done)
            obs     = next_obs
            ep_pnl += reward

            # Train after warmup
            if ep > config.WARMUP_EPISODES:
                agent.train_step()

        history["train_pnl"].append(float(info["pnl"]))


        if ep % config.EVAL_EVERY == 0 or ep == total_episodes:
            metrics = evaluate_agent(agent, eval_paths_S, eval_paths_v,
                                     tc=tc, n_episodes=300)
            history["episode"].append(ep)
            history["eval_mean"].append(metrics["mean"])
            history["eval_std"].append(metrics["std"])
            history["eval_cvar95"].append(metrics["cvar95"])
            history["wall_time"].append(time.time() - t0)

            # Average recent losses
            a_loss = float(np.mean(agent.actor_losses[-500:]))  if agent.actor_losses  else 0.0
            c_loss = float(np.mean(agent.critic_losses[-500:])) if agent.critic_losses else 0.0
            history["actor_loss"].append(a_loss)
            history["critic_loss"].append(c_loss)

            pbar.set_postfix({
                "eval_mean": f"{metrics['mean']:.4f}",
                "eval_std":  f"{metrics['std']:.4f}",
                "CVaR95":    f"{metrics['cvar95']:.4f}",
            })


        if ep % config.SAVE_EVERY == 0:
            agent.save(config.CHECKPOINT_DIR, tag=f"{run_name}_ep{ep}")

    # Final save
    agent.save(config.CHECKPOINT_DIR, tag=f"{run_name}_final")

    # Save history
    hist_path = os.path.join(config.RESULTS_DIR, f"{run_name}_history.json")
    with open(hist_path, "w") as f:
        json.dump(history, f, indent=2)
    print(f"\n[{agent.name}] History saved → {hist_path}")

    return history




def main():
    args = parse_args()
    np.random.seed(args.seed)

    print("\n" + "="*60)
    print("  EMPIRICAL DEEP HEDGING")
    print(f"  Model: {args.model.upper()}  |  TC: {args.tc*100:.1f}bps  |  Device: {args.device}")
    print("="*60)


    print("\n[Step 1/3] Generating training paths...")
    train_S, train_v = generate_and_cache(args.model, config.N_TRAIN_PATHS, seed=args.seed)

    print("[Step 1/3] Generating evaluation paths...")
    eval_S, eval_v   = generate_and_cache(args.model, config.N_EVAL_PATHS,  seed=args.seed+1)


    train_env = HedgingEnv(paths_S=train_S, paths_v=train_v, tc=args.tc)
    print(f"[Step 2/3] Env ready  — obs_space={train_env.observation_space.shape}, "
          f"action_space={train_env.action_space.shape}")


    print("[Step 3/3] Starting training...\n")
    histories = {}

    if args.agent in ("td3", "both"):
        td3 = TD3Agent(device=args.device)
        histories["TD3"] = train_agent(
            td3, train_env, eval_S, eval_v,
            total_episodes=args.episodes, tc=args.tc, run_name="td3"
        )

    if args.agent in ("ddpg", "both"):
        ddpg = DDPGAgent(device=args.device)
        histories["DDPG"] = train_agent(
            ddpg, train_env, eval_S, eval_v,
            total_episodes=args.episodes, tc=args.tc, run_name="ddpg"
        )


    print("\n" + "="*60)
    print("  TRAINING COMPLETE — Final Evaluation Metrics")
    print("="*60)
    for name, h in histories.items():
        if h["eval_mean"]:
            print(f"  {name:>5}  |  mean_PnL={h['eval_mean'][-1]:.4f}"
                  f"  |  std={h['eval_std'][-1]:.4f}"
                  f"  |  CVaR95={h['eval_cvar95'][-1]:.4f}")
    print("\nRun  python evaluate.py  to compare with Black-Scholes benchmark.")
    print("Run  python visualize.py  to generate result charts.\n")


if __name__ == "__main__":
    main()
