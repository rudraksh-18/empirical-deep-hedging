"""
visualize.py
"""

import argparse
import os, sys, json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns

import config
from data.generate_data       import HestonSimulator
from benchmarks.black_scholes import BlackScholesHedger, bs_delta


STYLE = {
    "figure.facecolor":  "#0d1117",
    "axes.facecolor":    "#161b22",
    "axes.edgecolor":    "#30363d",
    "axes.labelcolor":   "#c9d1d9",
    "xtick.color":       "#8b949e",
    "ytick.color":       "#8b949e",
    "text.color":        "#c9d1d9",
    "grid.color":        "#21262d",
    "grid.linestyle":    "--",
    "grid.alpha":        0.5,
    "legend.facecolor":  "#161b22",
    "legend.edgecolor":  "#30363d",
    "font.family":       "DejaVu Sans",
}
plt.rcParams.update(STYLE)

COLORS = {
    "TD3":           "#58a6ff",   # blue
    "DDPG":          "#ff7b72",   # red
    "Black-Scholes": "#3fb950",   # green
    "accent":        "#d2a8ff",   # purple
}


def save(fig, path):
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  Saved → {path}")




def plot_training_curves(save_dir):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Training Convergence: TD3 vs DDPG", fontsize=14, color="#c9d1d9", y=1.02)

    for agent, color in [("td3", COLORS["TD3"]), ("ddpg", COLORS["DDPG"])]:
        hist_path = os.path.join(config.RESULTS_DIR, f"{agent}_history.json")
        if not os.path.exists(hist_path):
            print(f"  [WARN] {hist_path} not found — skipping.")
            continue
        with open(hist_path) as f:
            h = json.load(f)

        eps  = h["episode"]
        mean = np.array(h["eval_mean"])
        std  = np.array(h["eval_std"])

        label = agent.upper()
        axes[0].plot(eps, mean, color=color, label=label, linewidth=2)
        axes[0].fill_between(eps, mean - std, mean + std, color=color, alpha=0.15)

        cvar = np.array(h["eval_cvar95"])
        axes[1].plot(eps, cvar, color=color, label=label, linewidth=2)

    for ax, title, ylabel in zip(
        axes,
        ["Mean Eval P&L over Training", "CVaR-95 (Risk) over Training"],
        ["Normalised P&L", "CVaR-95 (lower is better)"],
    ):
        ax.set_title(title, fontsize=11)
        ax.set_xlabel("Episode")
        ax.set_ylabel(ylabel)
        ax.legend()
        ax.grid(True)

    save(fig, os.path.join(save_dir, "training_curves.png"))




def plot_pnl_distribution(save_dir):
    eval_path = os.path.join(config.RESULTS_DIR, "eval_summary.json")
    if not os.path.exists(eval_path):
        print(f"  [WARN] {eval_path} not found. Run evaluate.py first.")
        return

    with open(eval_path) as f:
        summary = json.load(f)


    tc_key   = list(summary.keys())[0]
    results  = summary[tc_key]

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.set_title(f"P&L Distribution — {tc_key.replace('_', ' ').replace('tc ', 'TC=')}", fontsize=13)

    for r in results:
        label = r["label"]
        color = COLORS.get(label, "#c9d1d9")

        x = np.linspace(r["mean"] - 4 * r["std"], r["mean"] + 4 * r["std"], 300)
        from scipy.stats import norm as sn
        y = sn.pdf(x, loc=r["mean"], scale=r["std"])
        ax.plot(x, y, color=color, label=f"{label}  (μ={r['mean']:.3f}, σ={r['std']:.3f})",
                linewidth=2.5)
        ax.axvline(r["mean"], color=color, linestyle="--", alpha=0.6, linewidth=1.2)
        ax.fill_between(x, y, alpha=0.08, color=color)

    ax.set_xlabel("Normalised Episode P&L")
    ax.set_ylabel("Density")
    ax.legend(fontsize=10)
    ax.grid(True)
    save(fig, os.path.join(save_dir, "pnl_distribution.png"))




def plot_example_path(save_dir):
    sim = HestonSimulator()
    S, v = sim.simulate(n_paths=1, seed=7)
    S, v = S[0], v[0]

    t = np.arange(len(S))
    bs_deltas = [
        bs_delta(S[i], config.K, max((config.T_DAYS - i) * config.DT, 1e-6),
                 config.RISK_FREE, np.sqrt(max(v[i], 1e-8)))
        for i in range(len(S))
    ]

    fig = plt.figure(figsize=(14, 8))
    gs  = gridspec.GridSpec(2, 2, figure=fig, hspace=0.4, wspace=0.35)
    fig.suptitle("Example Heston Episode", fontsize=14, color="#c9d1d9")


    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(t, S, color=COLORS["TD3"], linewidth=1.8)
    ax1.axhline(config.K, color="#f0883e", linestyle="--", alpha=0.7, label=f"Strike K={config.K}")
    ax1.set_title("Underlying Price")
    ax1.set_ylabel("Price ($)")
    ax1.legend()
    ax1.grid(True)


    ax2 = fig.add_subplot(gs[0, 1])
    ax2.plot(t, np.sqrt(v) * 100, color=COLORS["DDPG"], linewidth=1.8)
    ax2.axhline(np.sqrt(config.THETA) * 100, color="#8b949e",
                linestyle="--", alpha=0.7, label=f"LR vol {np.sqrt(config.THETA)*100:.0f}%")
    ax2.set_title("Instantaneous Volatility")
    ax2.set_ylabel("Annualised Vol (%)")
    ax2.legend()
    ax2.grid(True)


    ax3 = fig.add_subplot(gs[1, 0])
    ax3.plot(t, bs_deltas, color=COLORS["Black-Scholes"], linewidth=1.8)
    ax3.set_title("Black-Scholes Delta (Hedge Ratio)")
    ax3.set_ylabel("Delta")
    ax3.set_xlabel("Trading Day")
    ax3.grid(True)


    ax4 = fig.add_subplot(gs[1, 1])
    log_rets = np.diff(np.log(S))
    ax4.hist(log_rets, bins=30, color=COLORS["accent"], edgecolor="#0d1117", alpha=0.85)
    ax4.set_title("Log-Return Distribution")
    ax4.set_xlabel("Log Return")
    ax4.set_ylabel("Count")
    ax4.grid(True)

    save(fig, os.path.join(save_dir, "example_path.png"))




def plot_tc_sweep(save_dir):
    eval_path = os.path.join(config.RESULTS_DIR, "eval_summary.json")
    if not os.path.exists(eval_path):
        print(f"  [WARN] {eval_path} not found. Run evaluate.py --tc 0.0 0.001 0.005 first.")
        return

    with open(eval_path) as f:
        summary = json.load(f)

    if len(summary) < 2:
        print("  [WARN] Only one TC level found. Run evaluate.py with multiple --tc values for this chart.")
        return

    tc_vals    = [float(k.replace("tc_", "")) * 10_000 for k in summary]  # in bps
    strategies = {r["label"]: [] for r in list(summary.values())[0]}

    for tc_key, results in summary.items():
        for r in results:
            strategies[r["label"]].append(r["cvar95"])

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.set_title("CVaR-95 vs Transaction Cost Rate", fontsize=13)
    for label, cvals in strategies.items():
        ax.plot(tc_vals, cvals, marker="o", color=COLORS.get(label, "#c9d1d9"),
                label=label, linewidth=2, markersize=7)

    ax.set_xlabel("Transaction Cost (bps)")
    ax.set_ylabel("CVaR-95 (↓ better)")
    ax.legend()
    ax.grid(True)
    save(fig, os.path.join(save_dir, "tc_sweep.png"))




def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--save_dir", type=str, default=os.path.join(config.RESULTS_DIR, "figs"))
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.save_dir, exist_ok=True)

    print("\n" + "="*50)
    print("  Generating visualizations...")
    print("="*50)

    plot_training_curves(args.save_dir)
    plot_pnl_distribution(args.save_dir)
    plot_example_path(args.save_dir)
    plot_tc_sweep(args.save_dir)

    print(f"\nAll charts saved to: {args.save_dir}/\n")


if __name__ == "__main__":
    main()
