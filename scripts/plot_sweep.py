#!/usr/bin/env python3
"""
Plot results from the hyperparameter sweep.
Generates comparison plots across all runs to identify which config produces grokking.
"""

import os
import sys
import json
import glob
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import numpy as np


def main():
    base_dir = os.path.join(os.path.dirname(__file__), '..')
    sweep_dir = os.path.join(base_dir, 'results', 'sweep')
    output_dir = os.path.join(base_dir, 'results', 'sweep_plots')
    os.makedirs(output_dir, exist_ok=True)

    # Find all completed runs
    config_files = sorted(glob.glob(os.path.join(sweep_dir, '*_config.json')))
    
    if not config_files:
        print("No sweep results found. Wait for runs to complete.")
        return

    runs = []
    for cf in config_files:
        with open(cf) as f:
            config = json.load(f)
        run_id = config['run_id']
        csv_path = os.path.join(sweep_dir, f'{run_id}_results.csv')
        if os.path.exists(csv_path):
            df = pd.read_csv(csv_path)
            runs.append((config, df))
            print(f"  Loaded {run_id}: {len(df)} checkpoints")
        else:
            print(f"  Skipping {run_id}: no results CSV yet")

    if not runs:
        print("No completed runs found yet.")
        return

    # ---- Plot 1: All runs comparison (Train + Test Loss) ----
    fig, axes = plt.subplots(len(runs), 1, figsize=(14, 5 * len(runs)), sharex=True)
    if len(runs) == 1:
        axes = [axes]
    
    fig.suptitle('Grokking Sweep: Train vs Test Loss per Configuration', fontsize=16, y=1.01)
    
    for idx, (config, df) in enumerate(runs):
        ax = axes[idx]
        ax.semilogy(df['Epoch'], df['Train_Loss'], label='Train Loss', color='#2196F3', lw=2)
        ax.semilogy(df['Epoch'], df['Test_Loss'], label='Test Loss', color='#FF9800', lw=2)
        
        title = (f"{config['run_id']}\n"
                 f"frac={config['frac_train']}, wd={config['weight_decay']}, "
                 f"init_scale={config['init_scale']}")
        ax.set_title(title, fontsize=11, fontweight='bold')
        ax.set_ylabel('Loss (log)')
        ax.legend(loc='upper right')
        ax.grid(True, alpha=0.3)
    
    axes[-1].set_xlabel('Epoch')
    plt.tight_layout()
    path = os.path.join(output_dir, 'sweep_loss_comparison.png')
    plt.savefig(path, dpi=200, bbox_inches='tight')
    print(f"\nSaved: {path}")
    plt.close()

    # ---- Plot 2: Overlay all runs on same axes ----
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), sharex=True)
    fig.suptitle('All Sweep Runs Overlaid', fontsize=16)
    
    colors = plt.cm.tab10(np.linspace(0, 1, len(runs)))
    
    for idx, (config, df) in enumerate(runs):
        label = (f"f={config['frac_train']}, wd={config['weight_decay']}, "
                 f"init={config['init_scale']}")
        ax1.semilogy(df['Epoch'], df['Train_Loss'], color=colors[idx], lw=1.5, 
                     label=f'{label} (train)', linestyle='-')
        ax2.semilogy(df['Epoch'], df['Test_Loss'], color=colors[idx], lw=1.5,
                     label=f'{label} (test)', linestyle='-')
    
    ax1.set_ylabel('Train Loss (log)')
    ax1.set_title('Train Loss')
    ax1.legend(fontsize=8, loc='upper right')
    ax1.grid(True, alpha=0.3)
    
    ax2.set_ylabel('Test Loss (log)')
    ax2.set_title('Test Loss')
    ax2.set_xlabel('Epoch')
    ax2.legend(fontsize=8, loc='upper right')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    path = os.path.join(output_dir, 'sweep_overlay.png')
    plt.savefig(path, dpi=200, bbox_inches='tight')
    print(f"Saved: {path}")
    plt.close()

    # ---- Plot 3: LLC and Order Parameter per run ----
    fig, axes = plt.subplots(len(runs), 2, figsize=(14, 4 * len(runs)))
    if len(runs) == 1:
        axes = axes.reshape(1, -1)
    
    fig.suptitle('LLC (Entropy) and Order Parameter per Configuration', fontsize=16, y=1.01)
    
    for idx, (config, df) in enumerate(runs):
        label = (f"f={config['frac_train']}, wd={config['weight_decay']}, "
                 f"init={config['init_scale']}")
        
        # LLC
        ax = axes[idx, 0]
        ax.plot(df['Epoch'], df['LLC'], color='purple', lw=1.5)
        ax.set_ylabel('LLC')
        ax.set_title(f'{label} — LLC', fontsize=9)
        ax.grid(True, alpha=0.3)
        
        # Order Parameter
        ax = axes[idx, 1]
        ax.plot(df['Epoch'], df['Order_Parameter'], color='green', lw=1.5)
        ax.set_ylabel('Order Param')
        ax.set_title(f'{label} — Order Param', fontsize=9)
        ax.grid(True, alpha=0.3)
    
    axes[-1, 0].set_xlabel('Epoch')
    axes[-1, 1].set_xlabel('Epoch')
    plt.tight_layout()
    path = os.path.join(output_dir, 'sweep_physics_metrics.png')
    plt.savefig(path, dpi=200, bbox_inches='tight')
    print(f"Saved: {path}")
    plt.close()

    # ---- Summary table ----
    print("\n" + "=" * 90)
    print("SWEEP SUMMARY")
    print("=" * 90)
    print(f"{'Run ID':<30} {'frac':>5} {'wd':>7} {'init':>5} "
          f"{'TrainLoss@5k':>13} {'TestLoss@5k':>13} {'TestLoss@end':>13}")
    print("-" * 90)
    
    for config, df in runs:
        # Get metrics at epoch 5000 and final epoch
        row_5k = df[df['Epoch'] <= 5000].iloc[-1] if len(df[df['Epoch'] <= 5000]) > 0 else None
        row_end = df.iloc[-1]
        
        train_5k = f"{row_5k['Train_Loss']:.6f}" if row_5k is not None else "N/A"
        test_5k = f"{row_5k['Test_Loss']:.4f}" if row_5k is not None else "N/A"
        test_end = f"{row_end['Test_Loss']:.4f}"
        
        print(f"{config['run_id']:<30} {config['frac_train']:>5.2f} "
              f"{config['weight_decay']:>7.4f} {config['init_scale']:>5.2f} "
              f"{train_5k:>13} {test_5k:>13} {test_end:>13}")
    
    print("\nGrokking signal: Train Loss → 0 early, Test Loss → 0 LATE (delayed generalization)")
    print(f"\nPlots saved to: {output_dir}/")


if __name__ == '__main__':
    main()
