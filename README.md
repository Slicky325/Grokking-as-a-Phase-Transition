# Thermodynamic Grokking Analysis

Modular PyTorch codebase for analyzing the **grokking** phenomenon as a thermodynamic phase transition. We track two physics-inspired metrics during training on modular arithmetic tasks:

- **Local Learning Coefficient (LLC)** — computed via SGLD from `devinterp`. Acts as an entropy proxy for model complexity.
- **Fourier Order Parameter (*M*)** — magnitude of the dominant frequency in the token embedding DFT. Measures the emergence of "clock" representations (symmetry breaking).

Models are trained on modular arithmetic mod *P* (default P ∈ {53, 113}) across two tasks: Addition and Division.

---

## Project Structure

```
slt-lbp/
├── src/
│   ├── data.py            # Modular arithmetic datasets (Addition, Division, Multi-task)
│   ├── models.py          # SmallTransformer (no-LayerNorm) and SmallMLP with init_scale
│   ├── metrics.py         # LLC (SGLD wrapper) and Fourier order parameter
│   └── trainer.py         # AdamW training loop with logarithmic checkpointing
├── scripts/
│   ├── run_experiments.py # Parallelised multi-task × multi-seed × multi-P experiments
│   ├── run_sweep.py       # Hyperparameter sweep: frac_train × weight_decay × init_scale
│   ├── plot_results.py    # Plot grokking curves from run_experiments output
│   └── plot_sweep.py      # Comparative plots from run_sweep output
├── ref/
│   ├── Papers/                    # Reference papers (PDFs)
│   ├── progress-measures-paper/   # Reference: Nanda et al. transformer + Fourier analysis
│   └── devinterp/                 # Reference: SGLD-based LLC estimation + MLP grokking
├── results/                       # Default output directory for experiments and sweeps
├── pyproject.toml                 # Project metadata and dependencies (managed by uv)
└── .gitignore
```

---

## Architecture

The default model is a **SmallTransformer** with no LayerNorm, matching the architecture from Nanda et al. / devinterp references:

| Component | Details |
|-----------|---------|
| Token embedding | `nn.Embedding(vocab_size, 128)` — vocab = `P + 3` (numbers 0…P-1, `+`, `/`, `=`) |
| Positional encoding | Learned `nn.Embedding(max_len, 128)` |
| Transformer layers | 1× `_NoNormTransformerLayer` — self-attention + ReLU MLP, **no LayerNorm** |
| Attention | 4 heads, causal mask |
| FFN | 512 hidden units, ReLU |
| Output | Linear unembedding from last token position |

An optional `use_ln=True` flag restores standard pre-norm `TransformerEncoderLayer`. A `SmallMLP` (embedding → ReLU → linear) is also available.

---

## Key Hyperparameters

Calibrated from the reference implementations in `ref/`:

| Parameter | Value | Source |
|-----------|-------|--------|
| `weight_decay` | **1.0** | `progress-measures-paper` (critical for grokking) |
| `lr` | `1e-3` | both references |
| `betas` | `(0.9, 0.98)` | `progress-measures-paper` |
| `frac_train` | `0.3` | `progress-measures-paper` default |
| LR warmup | Linear over 10 steps | matches Nanda et al. |

> **Why weight_decay = 1.0?**  
> Strong L2 regularisation is the mechanism that drives the phase transition from memorisation to generalisation. Values below ~0.1 produce confident-but-wrong predictions on the test set (test loss ≫ 3.97 random-chance ceiling) and grokking never occurs.

---

## Setup

```bash
# Create and activate the environment
uv sync          # or: pip install torch pandas devinterp seaborn matplotlib tqdm

# Activate
source .venv/bin/activate
```

**Requirements** (from `pyproject.toml`):

- Python ≥ 3.12
- `torch ≥ 2.10.0`
- `devinterp ≥ 1.3.2`
- `pandas ≥ 3.0.1`
- `matplotlib ≥ 3.10.8`
- `seaborn ≥ 0.13.2`
- `tqdm ≥ 4.67.3`

---

## Usage

### Main Experiments (parallelised, multi-P)

Trains for **40,000 epochs** across **2 tasks × 3 seeds × 2 P-values** (12 experiments total). Uses `ProcessPoolExecutor` with `spawn` context for CUDA-safe parallelism. Full-batch gradient descent (batch_size = 10,000).

LLC is computed via SGLD at each log checkpoint (skipped before epoch 200). The Fourier order parameter is always tracked.

```bash
python scripts/run_experiments.py
```

**Current configuration** (edit `run_experiments.py` to change):

| Parameter | Value |
|-----------|-------|
| Tasks | `addition`, `division` |
| Seeds | `42, 43, 44` |
| P values | `53, 113` |
| Epochs | `40,000` |
| Batch size | `10,000` (full-batch GD) |

Output:
- Per-run CSVs: `results/{task}_s{seed}_p{P}.csv`
- Combined CSV: `results/grokking_thermo_data.csv`

Then plot:

```bash
python scripts/plot_results.py
```

Output: `results/plots/{task}_grokking_thermo.png` — 3-panel figure (loss, LLC, order parameter) averaged over seeds with ±1σ bands.

---

### Hyperparameter Sweep (addition task, no LLC)

Sweeps over `frac_train × weight_decay × init_scale` (48 configurations total) on the addition task for **25,000 epochs** with a single seed. LLC is skipped for speed; only the Fourier order parameter is tracked.

Default grid (edit `run_sweep.py` to change):

| Parameter | Values |
|-----------|--------|
| `frac_train` | 0.2, 0.3, 0.5, 0.8 |
| `weight_decay` | 0.1, 0.5, 1.0, 2.0 |
| `init_scale` | 0.5, 1.0, 2.0 |

```bash
python scripts/run_sweep.py
```

Already-completed runs are skipped automatically (safe to resume after interruption).

Then plot:

```bash
python scripts/plot_sweep.py
```

Output in `results/sweep_plots/`:
- `sweep_loss_comparison.png` — per-config train/test loss panels
- `sweep_overlay.png` — all runs overlaid on shared axes
- `sweep_physics_metrics.png` — LLC and order parameter per config

---

### Running Long Jobs (tmux)

```bash
tmux new -s grok
source .venv/bin/activate
python scripts/run_experiments.py   # or run_sweep.py
# Ctrl+B, D  to detach
tmux attach -t grok                 # re-attach later
```

---

## Logging Schedule

Metrics are logged at logarithmically-spaced checkpoints early in training, then linearly:

- **Epochs 1–1000:** at epochs `1, 2, 5, 10, 20, 50, 100, 200, 500, 1000`
- **Epochs 1001+:** every 100 epochs

---

## Output Format

`grokking_thermo_data.csv` columns:

| Column | Description |
|--------|-------------|
| `Task` | `addition` or `division` |
| `Seed` | Random seed (42, 43, 44) |
| `P` | Prime modulus (53 or 113) |
| `Epoch` | Checkpoint epoch |
| `Train_Loss` | Cross-entropy on training set |
| `Test_Loss` | Cross-entropy on held-out set |
| `LLC` | Local Learning Coefficient (SGLD estimate) |
| `Order_Parameter` | Fourier power concentration at dominant frequency |

**Grokking signal:** `Train_Loss → 0` early, `Test_Loss → 0` delayed (often thousands of epochs later).

---

## References

Reference codebases in `ref/`:

- **progress-measures-paper** — Nanda et al. transformer for modular arithmetic with Fourier analysis
- **devinterp** — SGLD-based LLC estimation for grokking dynamics in MLPs
- **Papers/** — Relevant research papers (arXiv: 2201.02177, 2301.05217, 2308.12108, 2603.01192)
