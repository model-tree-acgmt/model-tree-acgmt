# Model Tree — Coefficient-Guided Piecewise Linear Regression

This repository contains two algorithms and two accompanying papers for piecewise linear regression using coefficient-guided model trees.

## The two algorithms

### RHM — Recursive Hybrid Model (Paper 1)

A minimal linear model tree. At each internal node, RHM fits a Ridge regression and selects the feature with the largest absolute coefficient as the splitting axis. It then searches a percentile grid for the best split point, partitions the data, and recurses. Each leaf holds a local Ridge model. The result is interpretable: a small tree of binary rules, each leaf with its own linear equation.

**Key design choice:** coefficient magnitude replaces exhaustive feature search (as in CART/M5), reducing per-node cost from O(nd² × d) to O(nd²).

- Implementation: `code.py`
- Paper: `paper.tex` — *A Recursive Hybrid Model for Piecewise Linear Regression: Algorithm Design, Analysis, and Empirical Evaluation*

### ACGMT — Adaptive Coefficient-Guided Model Tree (Paper 2)

A direct improvement over RHM that addresses its brittleness under scaling, collinearity, and regime-dependent structure. Four enhancements:

1. **Top-k feature preselection** — screen k features by coefficient magnitude instead of committing to one. This is the dominant improvement.
2. **Node-wise standardization** — center and scale features per node before ranking coefficients. Helps when features have different scales.
3. **Coarse-to-fine adaptive split search** — refine the best coarse split point with a denser local grid.
4. **Validation-aware pruning** — bottom-up pruning using a held-out validation set to collapse subtrees that don't help.

- Implementation: `acgmt.py`
- Paper: `second_paper.tex` — *Adaptive Coefficient-Guided Model Trees: Improving the Recursive Hybrid Model with Robust Split Selection*

## Results

Compared against Ridge, CART, Random Forest, Gradient Boosting, and RHM across five synthetic datasets. Averaged over 5 seeds.

| Dataset     | Ridge | CART  | RHM   | GradientBoosting | ACGMT (full) |
|-------------|-------|-------|-------|------------------|--------------|
| correlated  | 0.953 | 0.815 | 0.982 | 0.978            | 0.983        |
| interaction | 0.377 | 0.762 | 0.421 | 0.957            | 0.953        |
| irrelevant  | 0.954 | 0.810 | 0.992 | 0.988            | 0.992        |
| regime      | 0.270 | 0.749 | 0.261 | 0.928            | 0.914        |
| scaling     | 0.684 | 0.482 | 0.853 | 0.970            | 0.854        |

*Test R² (higher is better). See `results/summary.txt` for full tables including MSE and fit times.*

The biggest gains from RHM to ACGMT are on interaction (0.421 → 0.953) and regime (0.261 → 0.914) datasets, where top-1 feature selection fails but top-k finds the right split axis.

## Repository structure

```
code.py               # RHM implementation (Paper 1)
acgmt.py              # ACGMT implementation (Paper 2)
datasets.py           # Five synthetic dataset generators
experiments.py        # Main comparison: all models × all datasets
ablations.py          # Ablation studies for each ACGMT enhancement
utils.py              # Evaluation metrics and timing utilities
generate_summary.py   # Generates results/summary.txt from CSVs
paper.tex             # Paper 1: RHM
second_paper.tex      # Paper 2: ACGMT
results/              # Experiment output (CSVs + summary)
```

## Requirements

- Python 3.8+
- NumPy, pandas, scikit-learn
- matplotlib (optional, for plots)

```bash
pip install numpy pandas scikit-learn matplotlib
```

## Quick start

```python
# RHM (Paper 1)
from code import RecursiveHybridModel
model = RecursiveHybridModel(max_depth=3, min_samples_split=100)
model.fit(X_train, y_train)
predictions = model.predict(X_test)

# ACGMT (Paper 2)
from acgmt import AdaptiveCoefficientGuidedModelTree
model = AdaptiveCoefficientGuidedModelTree(max_depth=4, min_samples_split=100)
model.fit(X_train, y_train)
predictions = model.predict(X_test)
```

See [GUIDE.md](GUIDE.md) for detailed usage, parameter tuning, and applying to your own datasets.

## License

MIT
