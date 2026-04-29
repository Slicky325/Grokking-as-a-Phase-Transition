import torch
import torch.nn as nn
try:
    from devinterp.slt.sampler import estimate_learning_coeff as estimate_llc
except ImportError:
    from devinterp.slt import estimate_llc

def compute_llc(model, dataloader, criterion, num_chains=5, num_draws=200):
    """
    Wrapper function to compute the Local Learning Coefficient (LLC) using SGLD.
    """
    device = next(model.parameters()).device
    
    def evaluate(mod, batch):
        x, y = batch
        x, y = x.to(device), y.to(device)
        return criterion(mod(x), y)

    try:
        llc = estimate_llc(
            model=model,
            loader=dataloader,
            evaluate=evaluate,
            optimizer_kwargs=dict(lr=1e-5, localization=500.0),
            num_chains=num_chains,
            num_draws=num_draws,
            device=device,
            verbose=False,
        )
    except TypeError:
        # Fallback if the signature relies on criterion directly
        llc = estimate_llc(
            model=model,
            loader=dataloader,
            criterion=criterion,
            optimizer_kwargs=dict(lr=1e-5, localization=500.0),
            num_chains=num_chains,
            num_draws=num_draws,
            device=device,
            verbose=False,
        )
    return llc

def compute_order_parameter(model, P=53):
    """
    Measures how strongly the embedding matrix has developed 'clock' (Fourier)
    structure at any single dominant frequency.

    Method (matching ref/devinterp/modular_addition/dynamics.py::get_magnitude_modes):
      1. Compute the real DFT of W_E (shape P × d_model) along the token axis
         using explicit cosine/sine matrices — same as the reference codebase.
      2. Collect the magnitude of each frequency k across all embedding dims.
      3. Return the fraction of total Fourier POWER concentrated at the single
         most-energetic positive frequency (Herfindahl-style concentration index).

    Behaviour:
      - Random / pizza init:  ~1/(P//2) ≈ 0.04  (energy spread uniformly)
      - Grokked / clock:      → 0.3–0.9          (one key frequency dominates)
    """
    try:
        W = model.embedding.weight[:P, :].detach().float()
    except AttributeError:
        W = model.fc1.weight.T[:P, :].detach().float()

    device = W.device
    n = torch.arange(P, dtype=torch.float32, device=device)
    k = n.unsqueeze(1)                                      # (P, 1)
    cos_mat = torch.cos(2 * torch.pi * k * n / P)          # (P, P)
    sin_mat = torch.sin(2 * torch.pi * k * n / P)          # (P, P)

    c = cos_mat @ W                                         # (P, d_model)
    s = sin_mat @ W                                         # (P, d_model)
    mode_power = (c ** 2 + s ** 2).sum(dim=-1)             # (P,)  squared magnitude per freq

    # Positive frequencies only: k = 1 … P//2  (exclude DC k=0 and symmetric half)
    pos_power = mode_power[1 : P // 2 + 1]                 # (P//2,)

    # Concentration index: power at dominant freq / total positive-freq power
    # Bounded in [1/(P//2), 1]; increases monotonically as grokking sharpens one freq
    M = (pos_power.max() / pos_power.sum()).item()
    return M
