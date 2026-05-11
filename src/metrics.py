import math
import torch

from devinterp.slt.sampler import estimate_learning_coeff_with_summary


def compute_llc(model, dataloader, criterion,
                num_chains=5, num_draws=200,
                lr=1e-5, localization=500.0,
                cv_threshold=0.5, reject_negative=True):
    """
    Estimate the Local Learning Coefficient (LLC) via SGLD with chain diagnostics.

    Returns a dict:
        mean        : float — chain-averaged LLC (NaN if rejected)
        mean_raw    : float — chain-averaged LLC before acceptance check
        std         : float — std across chains
        cv          : float — std / |mean|  (sampler-disagreement proxy)
        per_chain   : list[float] — LLC value for each chain
        accepted    : bool — True if estimate passed quality gates
        num_chains  : int
        num_draws   : int

    Acceptance gates:
        - mean >= 0  (negative LLC indicates SGLD instability, not signal)
        - cv  <= cv_threshold  (chains disagree by < threshold * |mean|)
    """
    device = next(model.parameters()).device

    def evaluate(mod, batch):
        x, y = batch
        x, y = x.to(device), y.to(device)
        return criterion(mod(x), y)

    res = estimate_learning_coeff_with_summary(
        model=model,
        loader=dataloader,
        evaluate=evaluate,
        optimizer_kwargs=dict(lr=lr, localization=localization),
        num_chains=num_chains,
        num_draws=num_draws,
        device=device,
        verbose=False,
    )

    mean_raw = float(res["llc/mean"])
    std = float(res["llc/std"])
    per_chain = [float(res[f"llc-chain/{i}"]) for i in range(num_chains)]
    cv = std / abs(mean_raw) if abs(mean_raw) > 1e-9 else float("inf")

    accepted = True
    if reject_negative and mean_raw < 0:
        accepted = False
    if cv > cv_threshold:
        accepted = False

    return {
        "mean": mean_raw if accepted else float("nan"),
        "mean_raw": mean_raw,
        "std": std,
        "cv": cv,
        "per_chain": per_chain,
        "accepted": accepted,
        "num_chains": num_chains,
        "num_draws": num_draws,
    }


def compute_order_parameter(model, P=53):
    """
    Fourier-power concentration index on the embedding matrix.

    Random / pizza init: ~ 1 / (P//2)   (energy uniform across modes)
    Grokked / clock:     0.3 – 0.9       (one frequency dominates)
    """
    try:
        W = model.embedding.weight[:P, :].detach().float()
    except AttributeError:
        W = model.fc1.weight.T[:P, :].detach().float()

    device = W.device
    n = torch.arange(P, dtype=torch.float32, device=device)
    k = n.unsqueeze(1)
    cos_mat = torch.cos(2 * torch.pi * k * n / P)
    sin_mat = torch.sin(2 * torch.pi * k * n / P)

    c = cos_mat @ W
    s = sin_mat @ W
    mode_power = (c ** 2 + s ** 2).sum(dim=-1)

    pos_power = mode_power[1 : P // 2 + 1]
    return (pos_power.max() / pos_power.sum()).item()
