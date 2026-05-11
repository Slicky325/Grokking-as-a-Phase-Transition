#!/usr/bin/env python3
"""
Free-energy decomposition for grokking trajectories.

For each per-seed CSV in results/, plot:
    Energy term      E_n  = n * L_train         (empirical loss × training size)
    Complexity term  C_n  = λ * log(n)          (LLC × log size)
    Free energy      F_n  = E_n + C_n           (Bayesian free-energy expansion)

The point of the figure is to show how the contribution from L vs from λ shifts
across the grokking transition — the SLT story is that the complexity term picks
up the slack as the energy term falls.

Inputs are per-seed CSVs (one row per checkpoint) with at minimum:
    Epoch, Train_Loss, Test_Loss, LLC, Order_Parameter, Seed, Task, P

The training-set size n is inferred from (P, Task) and frac_train=0.3 to match
scripts/run_experiments.py — override with --n if needed.

Usage:
    python scripts/free_energy.py                # all CSVs in results/
    python scripts/free_energy.py --pattern 'addition_*.csv'
    python scripts/free_energy.py --frac 0.3 --P 113
"""

import argparse
import glob
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def infer_n(P: int, task: str, frac_train: float = 0.3) -> int:
    """Training-set size matching src/data.py."""
    if task == "addition":
        total = P * P
    elif task == "division":
        total = P * (P - 1)
    elif task == "multi-task":
        total = P * P + P * (P - 1)
    else:
        raise ValueError(f"unknown task {task!r}")
    return int(frac_train * total)


def smooth_llc(series: pd.Series, window: int = 5) -> pd.Series:
    """Median filter to suppress SGLD spikes while preserving the transition."""
    return series.rolling(window, center=True, min_periods=1).median()


def plot_one(df: pd.DataFrame, n: int, label: str, out_path: str) -> None:
    df = df.copy()
    df["LLC_smooth"] = smooth_llc(df["LLC"])

    # Energy and complexity in nats. Train_Loss is mean-CE → multiply by n for total NLL.
    df["E"] = n * df["Train_Loss"]
    df["C"] = df["LLC_smooth"] * np.log(n)
    df["F"] = df["E"] + df["C"]

    fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    fig.suptitle(f"Free-energy decomposition — {label}   (n={n})", fontsize=13)

    ax = axes[0]
    ax.plot(df["Epoch"], df["Train_Loss"], color="#1f77b4", lw=1.5, label="Train loss")
    ax.plot(df["Epoch"], df["Test_Loss"],  color="#d62728", lw=1.5, label="Test loss")
    ax.set_yscale("log")
    ax.set_ylabel("Cross-entropy loss")
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.3)
    ax.set_title("Learning curves")

    ax = axes[1]
    ax.plot(df["Epoch"], df["E"], color="#1f77b4", lw=1.8, label=r"Energy   $E_n = n\,L_n$")
    ax.plot(df["Epoch"], df["C"], color="#9467bd", lw=1.8, label=r"Complexity   $C_n = \lambda\,\log n$")
    ax.plot(df["Epoch"], df["F"], color="black",   lw=2.2, label=r"Free energy   $F_n = E_n + C_n$",
            linestyle="--")
    ax.set_yscale("symlog", linthresh=1.0)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Free energy (nats)")
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.3)
    ax.set_title("Bayesian free-energy expansion")

    plt.tight_layout()
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  → {out_path}")


def plot_overlay(dfs: list, ns: list, labels: list, out_path: str) -> None:
    """Overlay free-energy curves across seeds on shared axes."""
    fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    fig.suptitle("Free-energy decomposition — all seeds overlaid", fontsize=13)

    colors = plt.cm.tab10(np.linspace(0, 1, len(dfs)))

    for df, n, label, c in zip(dfs, ns, labels, colors):
        df = df.copy()
        df["LLC_smooth"] = smooth_llc(df["LLC"])
        E = n * df["Train_Loss"]
        C = df["LLC_smooth"] * np.log(n)

        axes[0].plot(df["Epoch"], E, color=c, lw=1.4, label=f"{label}  E")
        axes[1].plot(df["Epoch"], C, color=c, lw=1.4, label=f"{label}  C")

    axes[0].set_yscale("symlog", linthresh=1.0)
    axes[0].set_ylabel(r"$E_n = n\,L_n$ (nats)")
    axes[0].legend(fontsize=8, loc="upper right", ncol=2)
    axes[0].grid(True, alpha=0.3)
    axes[0].set_title("Energy term")

    axes[1].set_ylabel(r"$C_n = \lambda\,\log n$ (nats)")
    axes[1].set_xlabel("Epoch")
    axes[1].legend(fontsize=8, loc="upper right", ncol=2)
    axes[1].grid(True, alpha=0.3)
    axes[1].set_title("Complexity term")

    plt.tight_layout()
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  → {out_path}")


def run(results_dir: str, pattern: str = "*_p*.csv",
        frac: float = 0.3, P: int | None = None, task: str | None = None) -> int:
    """Generate free-energy plots for every per-seed CSV in results_dir.
    Returns the number of CSVs processed.
    """
    plots_dir = os.path.join(results_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)

    csv_files = sorted(glob.glob(os.path.join(results_dir, pattern)))
    csv_files = [f for f in csv_files if "grokking_thermo_data" not in f]
    if not csv_files:
        print(f"No CSVs match {pattern!r} in {results_dir}")
        return 0

    dfs, ns, labels = [], [], []
    for f in csv_files:
        df = pd.read_csv(f)
        if "LLC" not in df.columns:
            print(f"  skip {os.path.basename(f)}: no LLC column")
            continue
        if P is not None:
            P_use = P
        elif "P" in df.columns:
            P_use = int(df["P"].iloc[0])
        else:
            P_use = 113

        if task is not None:
            task_use = task
        elif "Task" in df.columns:
            task_use = str(df["Task"].iloc[0])
        else:
            task_use = os.path.basename(f).split("_")[0]

        n = infer_n(P_use, task_use, frac)
        label = os.path.splitext(os.path.basename(f))[0]

        out_path = os.path.join(plots_dir, f"{label}_free_energy.png")
        plot_one(df, n, label, out_path)

        dfs.append(df)
        ns.append(n)
        labels.append(label)

    if len(dfs) > 1:
        plot_overlay(dfs, ns, labels, os.path.join(plots_dir, "free_energy_overlay.png"))

    return len(dfs)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--results", default=None,
                   help="results dir (default: <repo>/results)")
    p.add_argument("--pattern", default="*_p*.csv",
                   help="glob for per-seed CSV files inside results/")
    p.add_argument("--frac", type=float, default=0.3,
                   help="training-set fraction used at run time (default 0.3)")
    p.add_argument("--P", type=int, default=None,
                   help="override modulus P (default: parse from filename)")
    p.add_argument("--task", default=None,
                   help="override task name (default: parse from filename)")
    args = p.parse_args()

    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    results_dir = args.results or os.path.join(base_dir, "results")

    n = run(results_dir, pattern=args.pattern, frac=args.frac,
            P=args.P, task=args.task)
    if n == 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
