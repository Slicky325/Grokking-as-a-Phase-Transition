#!/usr/bin/env python3
"""
Parallelised multi-seed, multi-task, multi-P experiment orchestrator.

Runs up to MAX_WORKERS (task, seed, P) triples simultaneously using
ProcessPoolExecutor with the 'spawn' context (required for CUDA in subprocesses).

Each worker shows its own tqdm epoch bar.
The main process shows an overall completion bar.
Each worker saves its own per-experiment CSV immediately on completion.
"""

import os
import sys
import torch
import pandas as pd
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm

# Workers inherit this via module re-import on spawn
_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

TASKS      = ['addition']#,  'division', 'multi-task']
SEEDS      = [42, 43, 44] # 45, 46]
P_VALUES   = [113]
EPOCHS     = 100_000
LR         = 1e-3
WD         = 1.0           # canonical value from progress-measures-paper
# Full-batch GD: batch_size must exceed the largest training split.
# P=113 multi-task training split ≈ 7628 → 10_000 covers all (P, task) combinations.
BATCH_SIZE = 10_000

# Skip LLC for epochs below this — SGLD at epoch 1–200 is expensive and uninformative.
LLC_MIN_EPOCH = 200


def _worker(task: str, seed: int, p: int, device_str: str, out_dir: str) -> "pd.DataFrame | None":
    """
    Train a single (task, seed, P) experiment.
    Executed in a subprocess — must re-import everything.
    """
    import sys
    sys.path.insert(0, _PROJECT_ROOT)

    import torch
    import pandas as pd
    from tqdm import tqdm
    from src.data import get_dataloaders
    from src.models import SmallTransformer
    from src.trainer import train_model

    vocab_size = p + 3  # 0..p-1=numbers, p=op_add, p+1=op_div, p+2=eq

    torch.manual_seed(seed)
    device = torch.device(device_str)
    if device.type == 'cuda':
        torch.cuda.manual_seed_all(seed)

    train_loader, test_loader = get_dataloaders(
        P=p, task=task, batch_size=BATCH_SIZE, frac_train=0.3, seed=seed
    )
    model = SmallTransformer(
        vocab_size=vocab_size, d_model=128, nhead=4, num_layers=1,
        use_ln=False,  # no LayerNorm — matches Nanda et al. reference exactly
    ).to(device)

    try:
        df = train_model(
            model=model,
            train_loader=train_loader,
            test_loader=test_loader,
            epochs=EPOCHS,
            lr=LR,
            weight_decay=WD,
            P=p,
            llc_min_epoch=LLC_MIN_EPOCH,
            tqdm_desc=f"{task}/s{seed}/p{p}",
        )
        df['Task'] = task
        df['Seed'] = seed
        df['P']    = p

        fname = f"{task.replace('-', '_')}_s{seed}_p{p}.csv"
        df.to_csv(os.path.join(out_dir, fname), index=False)
        tqdm.write(f"  saved {fname}  ({len(df)} rows)")
        return df

    except Exception as exc:
        import traceback
        tqdm.write(f"  [{task}/s{seed}/p{p}] FAILED: {exc}")
        traceback.print_exc()
        return None


def _choose_workers() -> int:
    """Conservative worker count based on available VRAM (~1 GB per worker)."""
    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        vram_mb = props.total_memory / 1024 ** 2
        workers = max(1, min(int(vram_mb // 1000), 4))
        tqdm.write(f"GPU: {props.name}  {vram_mb:.0f} MB VRAM  → {workers} workers")
    else:
        workers = max(1, os.cpu_count() // 2)
        tqdm.write(f"CPU: {os.cpu_count()} cores → {workers} workers")
    return workers


def main():
    out_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'results'))
    os.makedirs(out_dir, exist_ok=True)

    device_str  = 'cuda' if torch.cuda.is_available() else 'cpu'
    max_workers = _choose_workers()

    configs = [(task, seed, p) for task in TASKS for seed in SEEDS for p in P_VALUES]
    tqdm.write(f"Scheduling {len(configs)} experiments, {max_workers} at a time\n")

    ctx = mp.get_context('spawn')  # CUDA requires 'spawn', not 'fork'

    all_dfs = []
    with ProcessPoolExecutor(max_workers=max_workers, mp_context=ctx) as pool:
        futures = {
            pool.submit(_worker, task, seed, p, device_str, out_dir): (task, seed, p)
            for task, seed, p in configs
        }

        overall = tqdm(
            as_completed(futures),
            total=len(futures),
            desc="overall",
            unit="exp",
            position=0,
            dynamic_ncols=True,
        )
        for fut in overall:
            task, seed, p = futures[fut]
            try:
                df = fut.result()
                if df is not None:
                    all_dfs.append(df)
                    overall.set_postfix(last=f"{task}/s{seed}/p{p} ✓")
                else:
                    overall.set_postfix(last=f"{task}/s{seed}/p{p} ✗")
            except Exception as exc:
                tqdm.write(f"  [{task}/s{seed}/p{p}] executor error: {exc}")

    if not all_dfs:
        tqdm.write("No results — check errors above.")
        return

    final = pd.concat(all_dfs, ignore_index=True)
    cols = ['Task', 'Seed', 'P', 'Epoch', 'Train_Loss', 'Test_Loss', 'LLC', 'Order_Parameter']
    final = final[[c for c in cols if c in final.columns]]
    out_path = os.path.join(out_dir, 'grokking_thermo_data.csv')
    final.to_csv(out_path, index=False)
    tqdm.write(f"\nAll done — {len(final)} rows → {out_path}")


if __name__ == '__main__':
    main()
