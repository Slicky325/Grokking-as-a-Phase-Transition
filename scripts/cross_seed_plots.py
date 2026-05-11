#!/usr/bin/env python3
"""
Cross-seed plots ported from notebooks/thermodynamic_analysis.ipynb so the
notebook is no longer needed to produce them.

For each (task, P) group of per-seed CSVs in results/, generates:
    {task}_p{P}_aligned_thermodynamic.png   — loss/LLC/M aligned at t_c
    {task}_p{P}_phase_portrait.png          — LLC vs Order Parameter (seed-invariant)

t_c is detected per seed as the first epoch where test_loss < TC_TRIGGER.
"""

import argparse
import glob
import os
import re
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.ndimage import median_filter


TC_TRIGGER = 1.0
T_WINDOW   = 30_000
SMOOTH_K   = 5
LLC_CLIP   = 800


_FNAME_RE = re.compile(r"(?P<task>[a-z\-]+)_s(?P<seed>\d+)_p(?P<P>\d+)\.csv")


def _parse_filename(path: str) -> tuple[str, int, int] | None:
    m = _FNAME_RE.match(os.path.basename(path))
    if not m:
        return None
    return m["task"], int(m["seed"]), int(m["P"])


def _detect_tc(df: pd.DataFrame, trigger: float = TC_TRIGGER) -> int | None:
    below = df[df["Test_Loss"] < trigger]
    return None if below.empty else int(below["Epoch"].iloc[0])


def _prep(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["LLC"] = pd.to_numeric(df["LLC"], errors="coerce")
    df.loc[df["LLC"] < 0, "LLC"] = np.nan
    df["LLC_clipped"] = df["LLC"].clip(upper=LLC_CLIP)
    return df


def _aligned_plot(group: list[tuple[int, pd.DataFrame, int]],
                  task: str, P: int, out_path: str) -> None:
    """3-panel: loss, LLC, OP — all aligned so t' = epoch - t_c."""
    fig, axes = plt.subplots(3, 1, figsize=(10, 11), sharex=True)
    fig.suptitle(f"t_c-aligned thermodynamic frame — {task}, P={P}",
                 fontsize=13, fontweight="bold")
    ax_loss, ax_llc, ax_M = axes

    colors = plt.cm.tab10(np.linspace(0, 1, len(group)))
    for (seed, df, tc), c in zip(group, colors):
        a = df.copy()
        a["t_prime"] = a["Epoch"] - tc
        win = a[(a["t_prime"] >= -T_WINDOW) & (a["t_prime"] <= T_WINDOW)]

        ax_loss.plot(win["t_prime"], win["Train_Loss"], color=c, lw=1.2, alpha=0.9,
                     label=f"s{seed} train (t_c={tc:,})")
        ax_loss.plot(win["t_prime"], win["Test_Loss"],  color=c, lw=1.2, alpha=0.9,
                     linestyle="--", label=f"s{seed} test")

        llc_win = win.dropna(subset=["LLC_clipped"])
        if len(llc_win) > SMOOTH_K:
            sm = median_filter(llc_win["LLC_clipped"].values, size=SMOOTH_K)
            ax_llc.plot(llc_win["t_prime"], sm, color=c, lw=2,
                        label=f"s{seed}")

        ax_M.plot(win["t_prime"], win["Order_Parameter"], color=c, lw=2,
                  label=f"s{seed}")

    for ax in axes:
        ax.axvline(0, color="black", lw=0.8, linestyle=":")
        ax.grid(True, alpha=0.3)

    ax_loss.set_yscale("log")
    ax_loss.set_ylabel("Cross-entropy loss")
    ax_loss.set_title("Learning curves")
    ax_loss.legend(fontsize=7, loc="upper right", ncol=2)

    ax_llc.set_ylabel(r"LLC  $\lambda$  (clipped at %d)" % LLC_CLIP)
    ax_llc.set_title("Structural complexity (entropy proxy)")
    ax_llc.legend(fontsize=8, loc="upper right")

    ax_M.set_xlabel(r"$t' = $ epoch $- t_c$")
    ax_M.set_ylabel(r"Order parameter  $M$")
    ax_M.set_title("Fourier order parameter (symmetry breaking)")
    ax_M.legend(fontsize=8, loc="upper left")

    plt.tight_layout()
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  → {out_path}")


def _phase_portrait(group: list[tuple[int, pd.DataFrame, int]],
                    task: str, P: int, out_path: str) -> None:
    """LLC vs Order Parameter — seed-invariant trajectory in phase space."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle(f"LLC vs Order Parameter — seed-invariant phase portrait — {task}, P={P}",
                 fontsize=12, fontweight="bold")

    colors = plt.cm.tab10(np.linspace(0, 1, len(group)))
    for (seed, df, _tc), c in zip(group, colors):
        d = df.dropna(subset=["LLC_clipped", "Order_Parameter"])
        order = np.argsort(d["Order_Parameter"].values)
        axes[0].scatter(d["Order_Parameter"].values[order],
                        d["LLC_clipped"].values[order],
                        s=8, color=c, alpha=0.4, label=f"s{seed}")

        # Binned median for the right panel
        bins = np.linspace(d["Order_Parameter"].min(),
                           d["Order_Parameter"].max(), 20)
        idx = np.digitize(d["Order_Parameter"].values, bins)
        x_med, y_med, y_lo, y_hi = [], [], [], []
        for k in range(1, len(bins)):
            mask = idx == k
            if mask.sum() < 2:
                continue
            x_med.append(0.5 * (bins[k - 1] + bins[k]))
            vals = d["LLC_clipped"].values[mask]
            y_med.append(np.median(vals))
            y_lo.append(np.percentile(vals, 25))
            y_hi.append(np.percentile(vals, 75))
        axes[1].plot(x_med, y_med, color=c, lw=2, label=f"s{seed}")
        axes[1].fill_between(x_med, y_lo, y_hi, color=c, alpha=0.2)

    for ax in axes:
        ax.set_xlabel(r"Order parameter  $M$")
        ax.set_ylabel(r"LLC  $\lambda$")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)
    axes[0].set_title("Raw scatter")
    axes[1].set_title("Binned median ± IQR")

    plt.tight_layout()
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  → {out_path}")


def run(results_dir: str, pattern: str = "*_s*_p*.csv") -> int:
    """Group per-seed CSVs by (task, P) and emit cross-seed plots per group."""
    plots_dir = os.path.join(results_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)

    csv_files = sorted(glob.glob(os.path.join(results_dir, pattern)))
    groups: dict[tuple[str, int], list] = defaultdict(list)
    for f in csv_files:
        parsed = _parse_filename(f)
        if parsed is None:
            continue
        task, seed, P = parsed
        df = pd.read_csv(f)
        if not {"Epoch", "Train_Loss", "Test_Loss", "LLC", "Order_Parameter"}.issubset(df.columns):
            continue
        df = _prep(df)
        tc = _detect_tc(df)
        if tc is None:
            print(f"  skip {os.path.basename(f)}: never groks (no t_c)")
            continue
        groups[(task, P)].append((seed, df, tc))

    if not groups:
        print(f"  no per-seed CSVs matched {pattern!r}")
        return 0

    n_groups = 0
    for (task, P), group in groups.items():
        if len(group) < 2:
            print(f"  skip ({task}, P={P}): only {len(group)} seed — need ≥2 for cross-seed")
            continue
        group.sort(key=lambda x: x[0])
        prefix = f"{task}_p{P}"
        _aligned_plot(group, task, P,
                      os.path.join(plots_dir, f"{prefix}_aligned_thermodynamic.png"))
        _phase_portrait(group, task, P,
                        os.path.join(plots_dir, f"{prefix}_phase_portrait.png"))
        n_groups += 1
    return n_groups


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--results", default=None,
                   help="results dir (default: <repo>/results)")
    p.add_argument("--pattern", default="*_s*_p*.csv",
                   help="glob for per-seed CSV files (default: '*_s*_p*.csv')")
    args = p.parse_args()
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    results_dir = args.results or os.path.join(base_dir, "results")
    run(results_dir, pattern=args.pattern)


if __name__ == "__main__":
    main()
