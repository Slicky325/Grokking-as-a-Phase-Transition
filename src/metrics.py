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
            optimizer_kwargs=dict(lr=1e-5),
            num_chains=num_chains,
            num_draws=num_draws,
            device=device
        )
    except TypeError:
        # Fallback if the signature relies on criterion directly
        llc = estimate_llc(
            model=model,
            loader=dataloader,
            criterion=criterion,
            optimizer_kwargs=dict(lr=1e-5),
            num_chains=num_chains,
            num_draws=num_draws,
            device=device
        )
    return llc

def compute_order_parameter(model, P=53):
    """
    Extracts the embedding weight matrix (rows 0 to P-1), applies a 1D DFT,
    and returns the normalized magnitude of the first fundamental frequency (k=1)
    and its symmetric pair (k=P-1).
    """
    try:
        W_1 = model.embedding.weight[:P, :]
    except AttributeError:
        # Fallback for alternative MLP architecture if used
        W_1 = model.fc1.weight.T[:P, :]
        
    fft_W = torch.fft.fft(W_1, dim=0)
    M = (fft_W[1].norm() + fft_W[P-1].norm()).item() / P
    
    return M
