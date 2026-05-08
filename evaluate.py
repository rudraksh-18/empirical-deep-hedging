"""
evaluate.py
"""

import argparse
import os, sys, json
import numpy as np

import config
from data.generate_data    import generate_and_cache
from benchmarks.black_scholes import BlackScholesHedger
from agents.td3_agent      import TD3Agent
from agents.ddpg_agent     import DDPGAgent
from envs.hedging_env      import HedgingEnv




def evaluate_rl(agent, eval_S, eval_v, tc, n_episodes=1000):
    """Roll out agent deterministically; return array of episode P&Ls."""
    env  = HedgingEnv(paths_S=eval_S, paths_v=eval_v, tc=tc, mode="eval")
    pnls = []
    for _ in range(n_episodes):
        obs, _ = env.reset()
        done   = False
        while not done:
            action          = agent.select_action(obs, add_noise=False)
            obs, _, done, _, info = env.step(action)
        pnls.append(info["pnl"])
    return np.array(pnls)




def summarise(pnls: np.ndarray, label: str) -> dict:
    sorted_p = np.sort(pnls)
    cutoff   = max(int(len(pnls) * (1 - config.CVAR_ALPHA)), 1)
    cvar     = -float(sorted_p[:cutoff].mean())
    sharpe   = float(pnls.mean() / (pnls.std() + 1e-9))
    return {
        "label":    label,
        "mean":     float(pnls.mean()),
        "std":      float(pnls.std()),
        "cvar95":   cvar,
        "sharpe":   sharpe,
        "min":      float(pnls.min()),
        "max":      float(pnls.max()),
    }




def print_table(results: list):
    header = f"{'Strategy':<12} {'Mean P&L':>10} {'Std':>8} {'CVaR-95':>10} {'Sharpe':>8}"
    print("\n" + "="*55)
    print(header)
    print("-"*55)
    for r in results:
        print(
            f"{r['label']:<12} {r['mean']:>10.4f} {r['std']:>8.4f} "
            f"{r['cvar95']:>10.4f} {r['sharpe']:>8.4f}"
        )
    print("="*55 + "\n")




def parse_args():
    p = argparse.ArgumentParser(description="Evaluate deep hedging strategies")
    p.add_argument("--model",      type=str,   default="heston",
                   choices=["heston", "merton"])
    p.add_argument("--n_episodes", type=int,   default=1000)
    p.add_argument("--tc",         type=float, nargs="+", default=[config.TC],
                   help="One or more TC values to sweep (e.g. 0.0 0.001 0.005)")
    p.add_argument("--device",     type=str,   default="cpu")
    p.add_argument("--ckpt_dir",   type=str,   default=config.CHECKPOINT_DIR)
    return p.parse_args()


def main():
    args = parse_args()

    print("\n" + "="*60)
    print("  EMPIRICAL DEEP HEDGING — EVALUATION")
    print("="*60)


    eval_S, eval_v = generate_and_cache(args.model, config.N_EVAL_PATHS, seed=999)

    all_results = {}

    for tc in args.tc:
        print(f"\n─── Transaction Cost: {tc*100:.2f}bps {'─'*30}")

        results = []

        bs_hedger = BlackScholesHedger(tc=tc)
        bs_metrics = bs_hedger.evaluate(eval_S[:args.n_episodes], eval_v[:args.n_episodes])
        results.append(summarise(bs_metrics["all_pnls"], "Black-Scholes"))

        td3_ckpt = os.path.join(args.ckpt_dir, "td3_final_actor.pt")
        if os.path.exists(td3_ckpt):
            print("  Evaluating TD3...")
            td3 = TD3Agent(device=args.device)
            td3.load(args.ckpt_dir, tag="td3_final")
            td3.actor.eval()
            pnls = evaluate_rl(td3, eval_S, eval_v, tc=tc, n_episodes=args.n_episodes)
            results.append(summarise(pnls, "TD3"))
        else:
            print(f"  [WARN] TD3 checkpoint not found at {td3_ckpt}. Run train.py first.")

        ddpg_ckpt = os.path.join(args.ckpt_dir, "ddpg_final_actor.pt")
        if os.path.exists(ddpg_ckpt):
            print("  Evaluating DDPG...")
            ddpg = DDPGAgent(device=args.device)
            ddpg.load(args.ckpt_dir, tag="ddpg_final")
            ddpg.actor.eval()
            pnls = evaluate_rl(ddpg, eval_S, eval_v, tc=tc, n_episodes=args.n_episodes)
            results.append(summarise(pnls, "DDPG"))
        else:
            print(f"  [WARN] DDPG checkpoint not found at {ddpg_ckpt}. Run train.py first.")

        print_table(results)
        all_results[f"tc_{tc}"] = results


    os.makedirs(config.RESULTS_DIR, exist_ok=True)
    out_path = os.path.join(config.RESULTS_DIR, "eval_summary.json")
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"Results saved → {out_path}")
    print("Run  python visualize.py  to generate charts.\n")


if __name__ == "__main__":
    main()
