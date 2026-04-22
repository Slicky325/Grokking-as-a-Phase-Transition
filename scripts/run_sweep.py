#!/usr/bin/env python3
"""
Hyperparameter sweep over frac_train, weight_decay, and init_scale.
Trains on the addition task (single seed) per configuration for 25k epochs.
LLC is skipped for speed; Fourier order parameter is always computed.

Parameter ranges are chosen around the reference values from:
  - progress-measures-paper: wd=1.0, frac=0.3
  - devinterp modular_addition: wd=0.0002, frac=0.8, with and without wd

Results per run are saved to:
  results/sweep/{run_id}_config.json
  results/sweep/{run_id}_results.csv

After the sweep, run:
  python scripts/plot_sweep.py
"""

import os
import sys
import json
import itertools
import torch
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.data import get_dataloaders
from src.models import SmallTransformer
from src.trainer import train_model

# ── Sweep grid ────────────────────────────────────────────────────────────────
# Values chosen to bracket the two reference implementations
FRAC_TRAIN_VALUES   = [0.2, 0.3, 0.5, 0.8]   # tight → generous train split
WEIGHT_DECAY_VALUES = [0.1, 0.5, 1.0, 2.0]   # moderate → strong regularisation
INIT_SCALE_VALUES   = [0.5, 1.0, 2.0]         # sub-standard → over-initialized
# ─────────────────────────────────────────────────────────────────────────────

TASK       = 'addition'
SEED       = 42
EPOCHS     = 25_000
P          = 53
LR         = 1e-3
VOCAB_SIZE = P + 3   # 0..P-1=numbers, P=op_add, P+1=op_div, P+2=eq


def run_id_str(frac, wd, init):
    """Compact deterministic run identifier matching existing naming convention."""
    frac_tag = f"f{int(frac*100):03d}"
    wd_tag   = f"wd{str(wd).replace('.', 'e')}"
    init_tag = f"i{str(init).replace('.', '')}"
    return f"{frac_tag}_{wd_tag}_{init_tag}"


def main():
    base_dir  = os.path.join(os.path.dirname(__file__), '..')
    sweep_dir = os.path.join(base_dir, 'results', 'sweep')
    os.makedirs(sweep_dir, exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    grid = list(itertools.product(FRAC_TRAIN_VALUES, WEIGHT_DECAY_VALUES, INIT_SCALE_VALUES))
    print(f"Total configurations: {len(grid)}\n")

    for run_num, (frac, wd, init) in enumerate(grid, start=1):
        rid = f"run{run_num:02d}_{run_id_str(frac, wd, init)}"

        # Skip if already completed
        csv_path    = os.path.join(sweep_dir, f'{rid}_results.csv')
        config_path = os.path.join(sweep_dir, f'{rid}_config.json')
        if os.path.exists(csv_path) and os.path.exists(config_path):
            print(f"[{run_num:02d}/{len(grid)}] SKIP (exists): {rid}")
            continue

        print(f"\n[{run_num:02d}/{len(grid)}] {rid}")
        print(f"  frac_train={frac}, weight_decay={wd}, init_scale={init}")

        config = {
            'run_id':       rid,
            'task':         TASK,
            'seed':         SEED,
            'epochs':       EPOCHS,
            'frac_train':   frac,
            'weight_decay': wd,
            'init_scale':   init,
            'lr':           LR,
            'P':            P,
        }

        torch.manual_seed(SEED)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(SEED)

        train_loader, test_loader = get_dataloaders(
            P=P, task=TASK, batch_size=512, frac_train=frac, seed=SEED
        )

        model = SmallTransformer(
            vocab_size=VOCAB_SIZE, d_model=128, nhead=4, num_layers=1,
            init_scale=init,
        ).to(device)

        try:
            df = train_model(
                model=model,
                train_loader=train_loader,
                test_loader=test_loader,
                epochs=EPOCHS,
                lr=LR,
                weight_decay=wd,
                P=P,
                skip_llc=True,   # LLC is too slow for a parameter sweep
            )
            df['Task']   = TASK
            df['Seed']   = SEED
            df['Run_ID'] = rid

            # Save per-run outputs
            with open(config_path, 'w') as f:
                json.dump(config, f, indent=2)
            df.to_csv(csv_path, index=False)
            print(f"  Saved → {csv_path}")

            # Quick grokking summary
            final_row = df.iloc[-1]
            print(f"  Final  train={final_row['Train_Loss']:.4f}  "
                  f"test={final_row['Test_Loss']:.4f}  "
                  f"order={final_row['Order_Parameter']:.4f}")

        except Exception as e:
            print(f"  ERROR: {e}")

    print(f"\nSweep complete. Results in: {sweep_dir}/")
    print("Plot with:  python scripts/plot_sweep.py")


if __name__ == '__main__':
    main()
