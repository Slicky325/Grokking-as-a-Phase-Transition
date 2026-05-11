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


def _is_near_transition(test_loss: float,
                        lo: float = 0.01,
                        hi: float = 10.0) -> bool:
    """Heuristic: test_loss in (lo, hi) → we are inside the grokking transition.
    Outside this band the model is either deep in memorization (high test_loss)
    or already grokked (very low test_loss), and a cheaper SGLD run suffices.
    """
    return lo < test_loss < hi


def train_model(model, train_loader, test_loader, epochs=20_000, lr=1e-3,
                weight_decay=1.0, P=53, skip_llc=False,
                llc_min_epoch=300,
                llc_chains_default=5,    llc_draws_default=200,
                llc_chains_near_tc=10,   llc_draws_near_tc=500,
                llc_cv_threshold=0.5,
                tqdm_desc="training") -> pd.DataFrame:
    """
    AdamW training loop with logarithmic checkpointing.

    LLC is estimated at every checkpoint where it is allowed (epoch >= llc_min_epoch).
    Sampler intensity is adaptive: heavier SGLD when test_loss suggests we are inside
    the grokking transition (where the LLC drop is the science). Estimates that fail
    chain-disagreement diagnostics (CV > llc_cv_threshold) or are negative are stored
    as NaN — the raw value, std, cv, and acceptance flag are logged alongside.
    """
    device    = next(model.parameters()).device
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay,
                      betas=(0.9, 0.98))
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

        llc_mean = float("nan")
        llc_raw  = float("nan")
        llc_std  = float("nan")
        llc_cv   = float("nan")
        llc_acc  = False
        llc_nc   = 0
        llc_nd   = 0

        if not skip_llc and epoch >= llc_min_epoch:
            near_tc = _is_near_transition(test_loss)
            nc = llc_chains_near_tc if near_tc else llc_chains_default
            nd = llc_draws_near_tc  if near_tc else llc_draws_default
            try:
                d = compute_llc(model, train_loader, criterion,
                                num_chains=nc, num_draws=nd,
                                cv_threshold=llc_cv_threshold)
                llc_mean = d["mean"]
                llc_raw  = d["mean_raw"]
                llc_std  = d["std"]
                llc_cv   = d["cv"]
                llc_acc  = d["accepted"]
                llc_nc   = d["num_chains"]
                llc_nd   = d["num_draws"]
            except Exception as exc:
                tqdm.write(f"  [{tqdm_desc}] LLC failed at epoch {epoch}: {exc}")

        try:
            order_param = compute_order_parameter(model, P=P)
        except Exception as exc:
            tqdm.write(f"  [{tqdm_desc}] OP failed at epoch {epoch}: {exc}")
            order_param = float("nan")

        pbar.set_postfix(
            ep=epoch,
            tr=f"{train_loss:.3f}",
            te=f"{test_loss:.3f}",
            llc=f"{llc_mean:.0f}" if not math.isnan(llc_mean) else "—",
            cv=f"{llc_cv:.2f}" if not math.isnan(llc_cv) else "—",
            M=f"{order_param:.3f}",
        )

        results.append({
            "Epoch":           epoch,
            "Train_Loss":      train_loss,
            "Test_Loss":       test_loss,
            "LLC":             llc_mean,        # NaN if rejected
            "LLC_raw":         llc_raw,         # unfiltered chain mean
            "LLC_std":         llc_std,
            "LLC_cv":          llc_cv,
            "LLC_accepted":    llc_acc,
            "LLC_num_chains":  llc_nc,
            "LLC_num_draws":   llc_nd,
            "Order_Parameter": order_param,
        })
        model.train()

    return pd.DataFrame(results)
