#!/usr/bin/env python3
"""
Critical-exponent fit for the LLC trajectory leading up to t_c.

Hypothesis (continuous transition):
    λ(t') - λ_min  =  C · (-t')^α      for  t' = epoch - t_c  <  0

We fit α per seed inside a chosen window  t' ∈ [-FIT_WINDOW, -FIT_MIN]  via
linear regression in log-log space.  The point of this script is to make
that fit *defensible*:

    1. Bootstrap (resample with replacement inside the window, N_BOOT=1000)
       to get a 95% CI on α and verify it does not straddle zero.

    2. Window sweep: refit α over a grid of (FIT_WINDOW, FIT_MIN) values
       and plot α as a heatmap per seed.  A "real" exponent should be roughly
       invariant across reasonable window choices; if α moves a lot, the fit
       is window-artifact and the claim is weak.

Inputs: per-seed CSVs in results/  (Epoch, Train_Loss, Test_Loss, LLC, ...).
Outputs (results/plots/):
    crit_alpha_bootstrap.png
    crit_alpha_window_sweep.png
    crit_alpha_summary.csv
"""

import argparse
import glob
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import linregress


# Defaults — match the existing notebook so numbers are comparable
TC_TRIGGER     = 1.0       # test_loss threshold defining t_c
FIT_WINDOW_DEF = 12_000    # epochs before t_c to include
FIT_MIN_DEF    = 300       # epochs immediately before t_c to *exclude* (transition noise)
N_BOOT         = 1000

WINDOW_GRID = [5_000, 8_000, 12_000, 16_000, 20_000]
MIN_GRID    = [100, 300, 1_000, 3_000, 6_000]


def detect_tc(df: pd.DataFrame, trigger: float = TC_TRIGGER) -> int | None:
    """First epoch where test_loss falls below `trigger`."""
    below = df[df["Test_Loss"] < trigger]
    if below.empty:
        return None
    return int(below["Epoch"].iloc[0])


def smooth_llc(series: pd.Series, window: int = 5) -> pd.Series:
    return series.rolling(window, center=True, min_periods=1).median()


def fit_alpha(t_prime: np.ndarray, llc: np.ndarray,
              lam_min: float | None = None) -> tuple[float, float, float]:
    """
    Fit log(λ - λ_min) = log C + α · log(-t')

    Returns (alpha, intercept_logC, lam_min_used).
    """
    if lam_min is None:
        lam_min = float(np.nanmin(llc))
    y = llc - lam_min
    # Drop points where adjusted LLC ≤ 0 (cannot take log)
    mask = (y > 1e-3) & (t_prime < 0) & np.isfinite(y) & np.isfinite(t_prime)
    if mask.sum() < 5:
        return float("nan"), float("nan"), lam_min
    x_log = np.log(-t_prime[mask])
    y_log = np.log(y[mask])
    res = linregress(x_log, y_log)
    return float(res.slope), float(res.intercept), lam_min


def bootstrap_alpha(t_prime: np.ndarray, llc: np.ndarray,
                    n_boot: int = N_BOOT,
                    rng: np.random.Generator | None = None
                    ) -> tuple[float, float, float, np.ndarray]:
    """Return (alpha_point, alpha_lo95, alpha_hi95, bootstrap_samples)."""
    if rng is None:
        rng = np.random.default_rng(0)
    alpha_point, _, lam_min = fit_alpha(t_prime, llc)

    samples = np.empty(n_boot)
    n = len(t_prime)
    for i in range(n_boot):
        idx = rng.integers(0, n, size=n)
        a, _, _ = fit_alpha(t_prime[idx], llc[idx], lam_min=lam_min)
        samples[i] = a
    samples = samples[np.isfinite(samples)]
    lo, hi = np.percentile(samples, [2.5, 97.5])
    return alpha_point, float(lo), float(hi), samples


def window_sweep(t_prime: np.ndarray, llc: np.ndarray,
                 windows=WINDOW_GRID, mins=MIN_GRID) -> np.ndarray:
    """α as a function of (FIT_WINDOW, FIT_MIN). Shape (len(windows), len(mins))."""
    grid = np.full((len(windows), len(mins)), np.nan)
    for i, W in enumerate(windows):
        for j, m in enumerate(mins):
            mask = (t_prime >= -W) & (t_prime <= -m)
            if mask.sum() < 5:
                continue
            a, _, _ = fit_alpha(t_prime[mask], llc[mask])
            grid[i, j] = a
    return grid


def load_seeds(results_dir: str, pattern: str) -> dict:
    """Return {label: df_with_tc_aligned}, dropping seeds that never grok."""
    files = sorted(glob.glob(os.path.join(results_dir, pattern)))
    files = [f for f in files if "grokking_thermo_data" not in f]
    out = {}
    for f in files:
        df = pd.read_csv(f)
        if "LLC" not in df.columns:
            continue
        tc = detect_tc(df)
        if tc is None:
            print(f"  skip {os.path.basename(f)}: never groks (no t_c)")
            continue
        df = df.copy()
        df["LLC_smooth"] = smooth_llc(df["LLC"])
        df = df.dropna(subset=["LLC_smooth"])
        df["t_prime"] = df["Epoch"] - tc
        # Optional: clip absurd LLC spikes that distort the lognormal fit
        df["LLC_smooth"] = df["LLC_smooth"].clip(upper=df["LLC_smooth"].quantile(0.99))
        label = os.path.splitext(os.path.basename(f))[0]
        out[label] = (df, tc)
    return out


def plot_bootstrap(rows: list, out_path: str) -> None:
    fig, axes = plt.subplots(1, len(rows), figsize=(4.5 * len(rows), 4), sharey=True)
    if len(rows) == 1:
        axes = [axes]
    fig.suptitle("Bootstrap distribution of α (95% CI)", fontsize=13)

    for ax, row in zip(axes, rows):
        label, alpha, lo, hi, samples = row
        ax.hist(samples, bins=40, color="#9467bd", alpha=0.7)
        ax.axvline(alpha, color="black", lw=2,    label=f"α = {alpha:.3f}")
        ax.axvline(lo,    color="red",   lw=1.2, linestyle="--",
                   label=f"95% CI [{lo:.3f}, {hi:.3f}]")
        ax.axvline(hi,    color="red",   lw=1.2, linestyle="--")
        ax.axvline(0,     color="gray",  lw=1)
        ax.set_title(label, fontsize=10)
        ax.set_xlabel("α")
        ax.legend(fontsize=8, loc="upper right")
        ax.grid(True, alpha=0.3)
    axes[0].set_ylabel("count")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  → {out_path}")


def plot_window_sweep(grids: dict, out_path: str) -> None:
    fig, axes = plt.subplots(1, len(grids), figsize=(4.5 * len(grids), 4), sharey=True)
    if len(grids) == 1:
        axes = [axes]
    fig.suptitle("α vs fit-window choice  —  invariance check", fontsize=13)

    all_vals = np.concatenate([g.flatten() for g in grids.values()])
    all_vals = all_vals[np.isfinite(all_vals)]
    if len(all_vals) == 0:
        vmin, vmax = -1, 1
    else:
        vmin = float(np.nanpercentile(all_vals, 5))
        vmax = float(np.nanpercentile(all_vals, 95))

    for ax, (label, grid) in zip(axes, grids.items()):
        im = ax.imshow(grid, aspect="auto", origin="lower",
                       cmap="viridis", vmin=vmin, vmax=vmax)
        ax.set_xticks(range(len(MIN_GRID)))
        ax.set_xticklabels([str(m) for m in MIN_GRID])
        ax.set_yticks(range(len(WINDOW_GRID)))
        ax.set_yticklabels([f"{w//1000}k" for w in WINDOW_GRID])
        ax.set_xlabel("FIT_MIN (exclude this many epochs near t_c)")
        ax.set_title(label, fontsize=10)
        for i in range(grid.shape[0]):
            for j in range(grid.shape[1]):
                v = grid[i, j]
                if np.isfinite(v):
                    ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                            color="white", fontsize=8)
    axes[0].set_ylabel("FIT_WINDOW")
    fig.colorbar(im, ax=axes, shrink=0.8, label="α")
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  → {out_path}")


def run(results_dir: str, pattern: str = "*_p*.csv",
        n_boot: int = N_BOOT,
        fit_window: int = FIT_WINDOW_DEF, fit_min: int = FIT_MIN_DEF) -> int:
    """Run α fit + bootstrap + window sweep for every seed CSV in results_dir.
    Returns the number of seeds analyzed.
    """
    plots_dir = os.path.join(results_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)

    loaded = load_seeds(results_dir, pattern)
    if not loaded:
        print("No seeds with detectable t_c found.")
        return 0

    rng = np.random.default_rng(0)
    rows = []
    grids = {}
    summary = []

    for label, (df, tc) in loaded.items():
        mask = (df["t_prime"] >= -fit_window) & (df["t_prime"] <= -fit_min)
        t_p = df.loc[mask, "t_prime"].values
        llc = df.loc[mask, "LLC_smooth"].values
        if len(t_p) < 10:
            print(f"  {label}: too few points ({len(t_p)}) for fit")
            continue

        alpha, lo, hi, samples = bootstrap_alpha(t_p, llc, n_boot=n_boot, rng=rng)
        rows.append((label, alpha, lo, hi, samples))
        grids[label] = window_sweep(df["t_prime"].values, df["LLC_smooth"].values)
        summary.append({
            "label": label, "t_c": tc,
            "alpha": alpha, "alpha_lo95": lo, "alpha_hi95": hi,
            "n_points": len(t_p),
        })
        print(f"  {label}: t_c={tc:>6d}  α={alpha:.3f}  95% CI [{lo:.3f}, {hi:.3f}]"
              f"  n={len(t_p)}")

    if not rows:
        return 0

    pd.DataFrame(summary).to_csv(
        os.path.join(plots_dir, "crit_alpha_summary.csv"), index=False
    )
    plot_bootstrap(rows, os.path.join(plots_dir, "crit_alpha_bootstrap.png"))
    plot_window_sweep(grids, os.path.join(plots_dir, "crit_alpha_window_sweep.png"))
    return len(rows)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--results", default=None,
                   help="results dir (default: <repo>/results)")
    p.add_argument("--pattern", default="*_p*.csv",
                   help="glob for per-seed CSV files inside results/")
    p.add_argument("--n-boot", type=int, default=N_BOOT)
    p.add_argument("--fit-window", type=int, default=FIT_WINDOW_DEF)
    p.add_argument("--fit-min",    type=int, default=FIT_MIN_DEF)
    args = p.parse_args()

    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    results_dir = args.results or os.path.join(base_dir, "results")

    run(results_dir, pattern=args.pattern, n_boot=args.n_boot,
        fit_window=args.fit_window, fit_min=args.fit_min)


if __name__ == "__main__":
    main()
