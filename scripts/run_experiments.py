#!/usr/bin/env python3
"""
Multi-seed, multi-task experiment orchestrator.
Trains a SmallTransformer on modular arithmetic for 3 tasks × 5 seeds,
logging physics metrics (LLC, Fourier order parameter) at logarithmic checkpoints.
Results saved to results/grokking_thermo_data.csv.
"""

import os
import sys
import torch
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.data import get_dataloaders
from src.models import SmallTransformer
from src.trainer import train_model


TASKS  = ['addition', 'division', 'multi-task']
SEEDS  = [42, 43, 44, 45, 46]
EPOCHS = 20_000
P      = 53
LR     = 1e-3
WD     = 1.0          # canonical value from progress-measures-paper
# Token layout: 0..P-1=numbers, P=op_add, P+1=op_div, P+2=eq → vocab_size = P+3
VOCAB_SIZE = P + 3


def main():
    out_dir = os.path.join(os.path.dirname(__file__), '..', 'results')
    os.makedirs(out_dir, exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    all_results = []

    for task in TASKS:
        for seed in SEEDS:
            print(f"\n{'='*60}")
            print(f"Task: {task}  |  Seed: {seed}")
            print('='*60)

            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)

            train_loader, test_loader = get_dataloaders(
                P=P, task=task, batch_size=512, frac_train=0.3, seed=seed
            )

            model = SmallTransformer(
                vocab_size=VOCAB_SIZE, d_model=128, nhead=4, num_layers=1
            ).to(device)

            try:
                df = train_model(
                    model=model,
                    train_loader=train_loader,
                    test_loader=test_loader,
                    epochs=EPOCHS,
                    lr=LR,
                    weight_decay=WD,
                    P=P,
                )
                df['Task'] = task
                df['Seed'] = seed
                all_results.append(df)
            except Exception as e:
                print(f"ERROR — task={task} seed={seed}: {e}")

    if all_results:
        final = pd.concat(all_results, ignore_index=True)
        cols = ['Task', 'Seed', 'Epoch', 'Train_Loss', 'Test_Loss', 'LLC', 'Order_Parameter']
        final = final[[c for c in cols if c in final.columns]]
        out_path = os.path.join(out_dir, 'grokking_thermo_data.csv')
        final.to_csv(out_path, index=False)
        print(f"\nSaved {len(final)} rows → {out_path}")
    else:
        print("No results generated.")


if __name__ == '__main__':
    main()
