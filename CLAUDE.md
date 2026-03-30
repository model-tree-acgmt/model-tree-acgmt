# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Research project implementing two coefficient-guided piecewise linear regression algorithms:

- **RHM** (Recursive Hybrid Model) in `code.py` — selects the top-1 Ridge coefficient as split feature, searches a 9-point percentile grid for split point, recurses.
- **ACGMT** (Adaptive Coefficient-Guided Model Tree) in `acgmt.py` — improves RHM with four enhancements: top-k feature preselection, node-wise standardization, coarse-to-fine adaptive split search, and validation-aware pruning.

Both follow the scikit-learn estimator API (`fit`, `predict`, `get_params`, `set_params`).

## Commands

```bash
# Install dependencies (no setup.py/pyproject.toml — flat scripts)
pip install numpy pandas scikit-learn matplotlib

# Run smoke tests
python code.py          # RHM on synthetic data, outputs metrics + CSVs
python acgmt.py         # ACGMT smoke test (3 configs, prints MSE)

# Run full experiment suite
python experiments.py        # 5 datasets x 5 seeds x 8 models -> results/comparison_*.csv
python ablations.py          # Ablation studies -> results/ablation_*.csv
python generate_summary.py   # Aggregate CSVs -> results/summary.txt

# RHM tree visualization (requires graphviz)
python code.py --auto_generate_png
```

## Architecture

- `code.py` — `RecursiveHybridModel` class + `HybridNode`. Tree built via `_build_tree()`. Row-by-row prediction.
- `acgmt.py` — `AdaptiveCoefficientGuidedModelTree` class + `ACGMTNode`. Key internal methods: `_find_best_split()` (coarse+fine grid search per feature), `_prune_tree()` / `_prune_node()` (bottom-up validation pruning), `_predict_bulk()` (vectorized prediction).
- `datasets.py` — Five synthetic dataset factories (`make_scaling_data`, `make_correlated_data`, `make_irrelevant_data`, `make_interaction_data`, `make_regime_data`) exported via `ALL_DATASETS` dict.
- `experiments.py` — Compares 8 models (Ridge, CART, RF, GBR, RHM, 3 ACGMT variants) across all datasets.
- `ablations.py` — Isolates effect of each ACGMT enhancement (top-k, standardization, adaptive grid, pruning).
- `utils.py` — `evaluate_model()`, `time_fit_predict()`, `results_to_dataframe()`, `save_results()`.
- `generate_summary.py` — Reads CSVs from `results/`, outputs pivot tables to `results/summary.txt`.

## Key Design Details

- Both models accept pandas DataFrames or numpy arrays; feature names are auto-extracted from DataFrames.
- ACGMT holds out `val_fraction` (default 0.2) of training data for pruning — the tree is built on the remaining 80%.
- ACGMT clears `build_indices` after construction to free pruning-related state.
- RHM's `code.py` has a `main()` entry point that generates synthetic data and produces plots/CSVs; ACGMT's `acgmt.py` has a `__main__` block for smoke testing.
- No test framework (pytest/unittest) is used — validation is via the smoke test scripts and experiment harnesses.
