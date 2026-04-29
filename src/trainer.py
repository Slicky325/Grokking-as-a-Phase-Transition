import math
import torch
import torch.nn as nn
from torch.optim import AdamW
import pandas as pd
from tqdm import tqdm

from .metrics import compute_llc, compute_order_parameter


def should_log(epoch: int) -> bool:
    if epoch <= 1000:
        return epoch in [1, 2, 5, 10, 20, 50, 100, 200, 500, 1000]
    return epoch % 100 == 0


def train_model(model, train_loader, test_loader, epochs=20_000, lr=1e-3,
                weight_decay=1.0, P=53, skip_llc=False,
                llc_min_epoch=300, tqdm_desc="training") -> pd.DataFrame:
    """
    AdamW training loop with logarithmic checkpointing.
    Displays a tqdm progress bar; postfix updated at each checkpoint.
    """
    device    = next(model.parameters()).device
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay,
                      betas=(0.9, 0.98))
    # Linear warmup over 10 optimiser steps, matching Nanda et al. reference.
    # With full-batch GD (1 step/epoch) this ramps over 10 epochs.
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lr_lambda=lambda step: min(step / 10, 1.0)
    )
    criterion = nn.CrossEntropyLoss()

    results = []

    model.train()
    pbar = tqdm(range(1, epochs + 1), desc=tqdm_desc, unit="ep",
                dynamic_ncols=True, leave=True)

    for epoch in pbar:
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            criterion(model(x), y).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()

        if not should_log(epoch):
            continue

        model.eval()

        with torch.no_grad():
            train_loss = sum(
                criterion(model(x.to(device)), y.to(device)).item() * x.size(0)
                for x, y in train_loader
            ) / len(train_loader.dataset)

            test_loss = sum(
                criterion(model(x.to(device)), y.to(device)).item() * x.size(0)
                for x, y in test_loader
            ) / len(test_loader.dataset)

        if skip_llc or epoch < llc_min_epoch:
            llc = float('nan')
        else:
            try:
                llc = compute_llc(model, train_loader, criterion)
            except Exception as exc:
                tqdm.write(f"  [{tqdm_desc}] LLC failed at epoch {epoch}: {exc}")
                llc = float('nan')

        try:
            order_param = compute_order_parameter(model, P=P)
        except Exception as exc:
            tqdm.write(f"  [{tqdm_desc}] OP failed at epoch {epoch}: {exc}")
            order_param = float('nan')

        pbar.set_postfix(
            ep=epoch,
            tr=f"{train_loss:.3f}",
            te=f"{test_loss:.3f}",
            llc=f"{llc:.0f}" if not math.isnan(llc) else "—",
            M=f"{order_param:.3f}",
        )

        results.append({
            'Epoch':           epoch,
            'Train_Loss':      train_loss,
            'Test_Loss':       test_loss,
            'LLC':             llc,
            'Order_Parameter': order_param,
        })
        model.train()

    return pd.DataFrame(results)
