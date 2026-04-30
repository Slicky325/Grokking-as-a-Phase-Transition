import os
import glob
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def plot_csv(csv_path, output_dir):
    """Plot a single CSV file and save the figure, using the filename as the label."""
    filename = os.path.basename(csv_path)
    label = os.path.splitext(filename)[0]  # e.g. "addition_s42_p113"

    print(f"Loading data from {csv_path}...")
    df = pd.read_csv(csv_path)

    # Determine numeric columns to aggregate
    numeric_cols = [c for c in ['Train_Loss', 'Test_Loss', 'LLC', 'Order_Parameter'] if c in df.columns]

    if 'Epoch' not in df.columns:
        print(f"  Skipping {filename}: no 'Epoch' column found.")
        return

    # If there are multiple seeds, aggregate; otherwise just use the data directly
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
        else:
            return agg_df[col], None

    # 1. Train and Test Loss
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

    # 2. Local Learning Coefficient (LLC)
    ax = axes[1]
    if 'LLC' in numeric_cols:
        mean, std = _get('LLC')
        ax.plot(epochs, mean, label='LLC (Entropy Proxy)', color='purple', lw=2)
        if std is not None:
            ax.fill_between(epochs, mean - std, mean + std, alpha=0.2, color='purple')
    ax.set_ylabel('LLC')
    ax.legend()
    ax.set_title('Structural Complexity (Entropy Proxy)')

    # 3. Order Parameter
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
    # Setup paths
    base_dir = os.path.join(os.path.dirname(__file__), '..')
    results_dir = os.path.join(base_dir, 'results')
    output_dir = os.path.join(results_dir, 'plots')
    os.makedirs(output_dir, exist_ok=True)

    # Discover all CSV files in the results directory
    csv_files = sorted(glob.glob(os.path.join(results_dir, '*.csv')))

    if not csv_files:
        print("No CSV files found in results/. Have you successfully run run_experiments.py?")
        return

    print(f"Found {len(csv_files)} CSV file(s):")
    for f in csv_files:
        print(f"  - {os.path.basename(f)}")
    print()

    for csv_path in csv_files:
        plot_csv(csv_path, output_dir)


if __name__ == "__main__":
    main()
