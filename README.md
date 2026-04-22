# Thermodynamic Grokking Analysis

Modular PyTorch codebase for analyzing the **grokking** phenomenon as a thermodynamic phase transition. We track two physics-inspired metrics during training on modular arithmetic tasks:

- **Local Learning Coefficient (LLC)** — computed via SGLD from `devinterp`. Acts as an entropy proxy for model complexity.
- **Fourier Order Parameter (*M*)** — magnitude of the fundamental frequency in the token embedding DFT. Measures the emergence of "clock" representations (symmetry breaking).

Models are trained on modular arithmetic mod *P = 53* across three tasks: Addition, Division, and Multi-task.

---

## Project Structure

```
slt-lbp/
├── src/
│   ├── data.py       # Modular arithmetic datasets (Addition, Division, Multi-task)
│   ├── models.py     # SmallTransformer and SmallMLP with init_scale support
│   ├── metrics.py    # LLC (SGLD wrapper) and Fourier order parameter
│   └── trainer.py    # Training loop with logarithmic checkpointing
├── scripts/
│   ├── run_experiments.py  # Multi-task × multi-seed main experiment (3 tasks × 5 seeds)
│   ├── run_sweep.py        # Hyperparameter sweep: frac_train × weight_decay × init_scale
│   ├── plot_results.py     # Plot grokking curves from run_experiments output
│   └── plot_sweep.py       # Comparative plots from run_sweep output
├── notebooks/
│   └── thermodynamic_analysis.ipynb
├── results/
│   ├── grokking_thermo_data.csv   # Output of run_experiments.py
│   ├── plots/                     # Output of plot_results.py
│   ├── sweep/                     # Per-run configs + CSVs from run_sweep.py
│   └── sweep_plots/               # Output of plot_sweep.py
└── ref/
    ├── progress-measures-paper/   # Reference: Nanda et al. transformer + Fourier analysis
    └── devinterp/                 # Reference: SGLD-based LLC estimation + MLP grokking
```

---

## Key Hyperparameters

These are calibrated from the reference implementations in `ref/`:

| Parameter | Value | Source |
|-----------|-------|--------|
| `weight_decay` | **1.0** | `progress-measures-paper` (critical for grokking) |
| `lr` | `1e-3` | both references |
| `betas` | `(0.9, 0.98)` | `progress-measures-paper` |
| `frac_train` | `0.3` | `progress-measures-paper` default |
| `epochs` | `20 000` | sufficient for grokking onset |

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

---

## Usage

### Main Experiments (3 tasks × 5 seeds, full LLC + order parameter)

Trains for 20 000 epochs, logs at logarithmic checkpoints, computes LLC via SGLD and Fourier order parameter at each checkpoint.

```bash
python scripts/run_experiments.py
```

Output: `results/grokking_thermo_data.csv`

Then plot:

```bash
python scripts/plot_results.py
```

Output: `results/plots/{task}_grokking_thermo.png` per task.

---

### Hyperparameter Sweep (addition task, no LLC)

Sweeps over `frac_train × weight_decay × init_scale` (48 configurations total). LLC is skipped for speed; only the Fourier order parameter is tracked.

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

Output: `results/sweep_plots/` — loss comparison, overlaid curves, physics metrics per config.

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

## Output Format

`grokking_thermo_data.csv` columns:

| Column | Description |
|--------|-------------|
| `Task` | `addition`, `division`, or `multi-task` |
| `Seed` | Random seed (42–46) |
| `Epoch` | Checkpoint epoch |
| `Train_Loss` | Cross-entropy on training set |
| `Test_Loss` | Cross-entropy on held-out set |
| `LLC` | Local Learning Coefficient (SGLD estimate) |
| `Order_Parameter` | Normalised DFT magnitude at *k* = 1 |

Grokking signal: `Train_Loss → 0` early, `Test_Loss → 0` delayed (often thousands of epochs later).
