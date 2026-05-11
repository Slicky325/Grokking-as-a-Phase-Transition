"""
Per-seed thermodynamic plots, plus free-energy decomposition and
critical-exponent analysis (steps 4 and 5).

Usage:
    python scripts/plot_results.py                       # uses ../results
    python scripts/plot_results.py --results results-1   # any results dir
"""

import argparse
import glob
import os
import sys

# Ensure sibling scripts (free_energy.py, critical_exponent.py) resolve regardless
# of which directory plot_results.py is launched from.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

import free_energy
import critical_exponent
import cross_seed_plots


def plot_csv(csv_path, output_dir):
    """Plot a single CSV file and save the figure, using the filename as the label."""
    filename = os.path.basename(csv_path)
    label = os.path.splitext(filename)[0]

    print(f"Loading data from {csv_path}...")
    df = pd.read_csv(csv_path)

    numeric_cols = [c for c in ['Train_Loss', 'Test_Loss', 'LLC', 'Order_Parameter'] if c in df.columns]

    if 'Epoch' not in df.columns:
        print(f"  Skipping {filename}: no 'Epoch' column found.")
        return

    if 'Seed' in df.columns and df['Seed'].nunique() > 1:
        agg_df = df.groupby('Epoch')[numeric_cols].agg(['mean', 'std'])
        has_std = True
    else:
        agg_df = df.set_index('Epoch')[numeric_cols]
        has_std = False

    epochs = agg_df.index
    sns.set_theme(style="whitegrid")
    fig, axes = plt.subplots(3, 1, figsize=(10, 15), sharex=True)
    fig.suptitle(f'Thermodynamic Grokking Analysis: {label}', fontsize=16)

    def _get(col):
        if has_std:
            return agg_df[col]['mean'], agg_df[col]['std']
        return agg_df[col], None

    ax = axes[0]
    for col, color in [('Train_Loss', 'blue'), ('Test_Loss', 'red')]:
        if col not in numeric_cols:
            continue
        mean, std = _get(col)
        ax.plot(epochs, mean, label=col.replace('_', ' '), color=color, lw=2)
        if std is not None:
            ax.fill_between(epochs, mean - std, mean + std, alpha=0.2, color=color)
    ax.set_ylabel('Cross-Entropy Loss')
    ax.set_yscale('log')
    ax.legend()
    ax.set_title('Learning Curves')

    ax = axes[1]
    if 'LLC' in numeric_cols:
        mean, std = _get('LLC')
        ax.plot(epochs, mean, label='LLC (Entropy Proxy)', color='purple', lw=2)
        if std is not None:
            ax.fill_between(epochs, mean - std, mean + std, alpha=0.2, color='purple')
    ax.set_ylabel('LLC')
    ax.legend()
    ax.set_title('Structural Complexity (Entropy Proxy)')

    ax = axes[2]
    if 'Order_Parameter' in numeric_cols:
        mean, std = _get('Order_Parameter')
        ax.plot(epochs, mean, label='Order Parameter', color='green', lw=2)
        if std is not None:
            ax.fill_between(epochs, mean - std, mean + std, alpha=0.2, color='green')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Magnitude (k=1)')
    ax.legend()
    ax.set_title('Fourier Order Parameter (Symmetry Breaking)')

    plt.tight_layout()
    plt.subplots_adjust(top=0.92)

    plot_path = os.path.join(output_dir, f'{label}_grokking_thermo.png')
    plt.savefig(plot_path, dpi=300)
    print(f"Saved {plot_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", default=None,
                        help="results dir (default: <repo>/results)")
    parser.add_argument("--pattern", default="*_s*_p*.csv",
                        help="glob for per-seed CSV files (default: '*_s*_p*.csv')")
    parser.add_argument("--skip-free-energy", action="store_true",
                        help="don't run free-energy analysis")
    parser.add_argument("--skip-critical-exponent", action="store_true",
                        help="don't run critical-exponent analysis")
    parser.add_argument("--skip-cross-seed", action="store_true",
                        help="don't run cross-seed aligned/phase plots")
    parser.add_argument("--n-boot", type=int, default=1000,
                        help="bootstrap samples for α CI (default 1000)")
    args = parser.parse_args()

    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    results_dir = args.results or os.path.join(base_dir, 'results')
    output_dir = os.path.join(results_dir, 'plots')
    os.makedirs(output_dir, exist_ok=True)

    # Per-seed CSVs only. Skips the aggregate `grokking_thermo_data.csv` —
    # plotting an averaged-across-everything CSV doesn't say anything useful.
    csv_files = sorted(glob.glob(os.path.join(results_dir, args.pattern)))

    if not csv_files:
        print(f"No CSVs match {args.pattern!r} in {results_dir}.")
        return

    print(f"Found {len(csv_files)} per-seed CSV(s):")
    for f in csv_files:
        print(f"  - {os.path.basename(f)}")
    print()

    for csv_path in csv_files:
        plot_csv(csv_path, output_dir)

    if not args.skip_cross_seed:
        print("\n=== Cross-seed aligned + phase-portrait plots ===")
        cross_seed_plots.run(results_dir, pattern=args.pattern)

    if not args.skip_free_energy:
        print("\n=== Free-energy decomposition ===")
        free_energy.run(results_dir, pattern=args.pattern)

    if not args.skip_critical_exponent:
        print("\n=== Critical-exponent analysis ===")
        critical_exponent.run(results_dir, pattern=args.pattern,
                              n_boot=args.n_boot)


if __name__ == "__main__":
    main()
