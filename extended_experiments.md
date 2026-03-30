# Extended Experiments: Synthetic and Real-World Dataset Evaluation

This document reports a comprehensive evaluation of RHM and ACGMT against standard baselines on 11 datasets (5 synthetic + 6 real-world/standard). All results are averaged over 5 random seeds (42, 123, 456, 789, 1024) with 80/20 train-test splits.

---

## Key Findings

1. **ACGMT consistently improves over RHM across all tested scenarios.** On real-world data, ACGMT (full) improves over RHM by up to +1.41 R² (California Housing) and averages +0.28 R² across the 6 real/standard datasets. The only exception is the yacht dataset (308 samples, exponential target), where both methods perform similarly and all linear-leaf models struggle.

2. **Top-k feature preselection is the single most impactful enhancement.** It prevents catastrophic failure on datasets where top-1 commits to the wrong split feature. The improvement is largest on interaction (+0.53 R²), regime (+0.65 R²), and California Housing (+1.41 R²). When top-1 already selects the right feature (correlated, irrelevant), gains are marginal.

3. **ACGMT approaches GradientBoosting accuracy on piecewise-linear problems while remaining interpretable.** On correlated (0.983 vs 0.978), irrelevant (0.992 vs 0.988), and interaction (0.953 vs 0.957), ACGMT matches or rivals GradientBoosting with a single depth-4 tree containing at most 16 inspectable linear equations.

4. **Pruning is beneficial on large datasets but counterproductive on small ones.** With fewer than ~1,000 training samples, the 20% validation holdout leaves too few samples for reliable pruning decisions. ACGMT (top3, std) without pruning outperforms ACGMT (full) on airfoil (0.756 vs 0.702) and concrete (0.844 vs 0.838).

5. **GradientBoosting dominates on real-world data.** It achieves the best test R² on 5 of 6 real/standard datasets, with the largest margins on yacht (0.998 vs 0.577) and California Housing (0.811 vs 0.696). This is expected — an ensemble of 100 additive depth-4 trees has far more capacity than a single depth-4 tree with linear leaves. The gap narrows on datasets with piecewise-linear structure (concrete: 0.923 vs 0.838) and reverses on synthetic datasets designed for linear-leaf models (correlated: 0.978 vs 0.983).

6. **Linear-leaf model trees have a fundamental limitation on exponential/highly-nonlinear targets.** On yacht (exponential Froude-resistance relationship) and scaling (quadratic component), constant-leaf methods (CART, RF, GBR) outperform all linear-leaf models by a wide margin. This is inherent to the model tree family, not specific to RHM/ACGMT.

7. **RHM can fail catastrophically.** On California Housing, RHM produces negative R² (-0.71 with std = 2.96) because the top-1 heuristic commits to a poor split feature on some seeds, creating degenerate tree structures. ACGMT's enhancements eliminate this failure mode entirely (R² = 0.696, std = 0.055).

---

## Table of Contents

- [Key Findings](#key-findings)
- [Models](#models)
- [Datasets](#datasets)
  - [Synthetic Datasets](#synthetic-datasets)
  - [Real-World Datasets](#real-world-datasets)
- [Results: Synthetic Datasets](#results-synthetic-datasets)
  - [Test R²](#test-r²-mean--std-over-5-seeds)
  - [Key Findings (Synthetic)](#key-findings-synthetic)
- [Results: Real-World Datasets](#results-real-world-datasets)
  - [Test R²](#test-r²-mean--std-over-5-seeds-1)
  - [Test MSE](#test-mse-mean--std-over-5-seeds)
  - [Test MAE](#test-mae-mean-over-5-seeds)
- [Overfitting Analysis](#overfitting-analysis)
- [Timing](#timing)
- [Per-Dataset Analysis](#per-dataset-analysis)
  - [California Housing](#california-housing-n20640-d8)
  - [Diabetes](#diabetes-n442-d10)
  - [Concrete Compressive Strength](#concrete-compressive-strength-n1030-d8)
  - [Airfoil Self-Noise](#airfoil-self-noise-n1503-d5)
  - [Yacht Hydrodynamics](#yacht-hydrodynamics-n308-d6)
  - [Friedman #1](#friedman-1-n5000-d10)
- [Cross-Cutting Findings](#cross-cutting-findings)
- [Reproducibility](#reproducibility)

---

## Models

| Model | Description |
|-------|-------------|
| **Ridge** | Linear regression with L2 regularization (alpha=1.0) |
| **CART** | Decision tree regressor (max_depth=4) |
| **RandomForest** | 100 trees, max_depth=4 |
| **GradientBoosting** | 100 boosting rounds, max_depth=4 |
| **RHM (original)** | Recursive Hybrid Model — top-1 coefficient, 9-point percentile grid, no standardization |
| **ACGMT (top1, no-std)** | ACGMT with top_k=1, no standardization, no adaptive grid, no pruning — equivalent to RHM logic in the ACGMT codebase |
| **ACGMT (top3, std)** | ACGMT with top_k=3, node-wise standardization, no adaptive grid, no pruning |
| **ACGMT (full)** | Full ACGMT: top_k=3, standardization, adaptive grid, validation-aware pruning (val_fraction=0.2) |

All tree-based models use max_depth=4 and min_samples_split=100 (for RHM/ACGMT).

---

## Datasets

### Synthetic Datasets

| Dataset | n_train | n_test | Features | Description |
|---------|---------|--------|----------|-------------|
| **scaling** | 4,000 | 1,000 | 5 | Features at scales 1x to 1000x; equal true contribution after rescaling. Tests scale invariance. |
| **correlated** | 4,000 | 1,000 | 5 | Correlated feature pairs (rho=0.95); only one per pair is predictive. Tests collinearity robustness. |
| **irrelevant** | 4,000 | 1,000 | 20 | 3 relevant features among 17 noise features. Tests feature selection under noise. |
| **interaction** | 4,000 | 1,000 | 5 | Regime shift at x0=0 with different interaction effects per regime. Tests piecewise structure discovery. |
| **regime** | 4,000 | 1,000 | 5 | Piecewise linear in x0 with 4 breakpoints and varying slopes/intercepts. Tests multi-regime recovery. |

### Real-World Datasets

| Dataset | n_train | n_test | Features | Source | Target Variable |
|---------|---------|--------|----------|--------|-----------------|
| **california** | 16,512 | 4,128 | 8 | sklearn built-in | Median house value ($100k units). Features: MedInc, HouseAge, AveRooms, AveBedrms, Population, AveOccup, Latitude, Longitude. |
| **diabetes** | 353 | 89 | 10 | sklearn built-in | Disease progression one year after baseline. 10 pre-normalized features (age, sex, bmi, bp, s1-s6). |
| **concrete** | 824 | 206 | 8 | OpenML (id=4353) | Concrete compressive strength (MPa). Features: cement, blast furnace slag, fly ash, water, superplasticizer, coarse aggregate, fine aggregate, age. |
| **airfoil** | 1,202 | 301 | 5 | OpenML | Scaled sound pressure level (dB). Features: frequency, angle, chord length, free-stream velocity, suction-side thickness. NASA aeroacoustic dataset. |
| **yacht** | 246 | 62 | 6 | OpenML | Residuary resistance per unit displacement. Features: longitudinal position, prismatic coefficient, length-displacement ratio, beam-draught ratio, length-beam ratio, Froude number. |
| **friedman1** | 4,000 | 1,000 | 10 | sklearn synthetic | y = 10*sin(pi*x1*x2) + 20*(x3-0.5)^2 + 10*x4 + 5*x5 + noise. 5 informative + 5 noise features. Standard model tree benchmark. |

---

## Results: Synthetic Datasets

### Test R² (mean +/- std over 5 seeds)

| Dataset | Ridge | CART | RandomForest | GradientBoosting | RHM | ACGMT (top1) | ACGMT (top3, std) | ACGMT (full) |
|---------|-------|------|--------------|------------------|-----|--------------|-------------------|--------------|
| scaling | 0.684 +/- 0.017 | 0.483 +/- 0.022 | 0.668 +/- 0.009 | **0.970 +/- 0.001** | 0.853 +/- 0.009 | 0.853 +/- 0.009 | 0.856 +/- 0.017 | 0.854 +/- 0.027 |
| correlated | 0.953 +/- 0.004 | 0.815 +/- 0.006 | 0.863 +/- 0.007 | 0.978 +/- 0.002 | 0.982 +/- 0.001 | 0.982 +/- 0.001 | 0.983 +/- 0.001 | **0.983 +/- 0.001** |
| irrelevant | 0.954 +/- 0.003 | 0.810 +/- 0.010 | 0.872 +/- 0.010 | 0.988 +/- 0.001 | 0.992 +/- 0.000 | 0.992 +/- 0.000 | **0.992 +/- 0.001** | 0.992 +/- 0.001 |
| interaction | 0.377 +/- 0.047 | 0.762 +/- 0.014 | 0.831 +/- 0.010 | **0.957 +/- 0.005** | 0.421 +/- 0.091 | 0.421 +/- 0.091 | 0.953 +/- 0.008 | 0.953 +/- 0.011 |
| regime | 0.270 +/- 0.026 | 0.749 +/- 0.009 | 0.783 +/- 0.007 | **0.928 +/- 0.005** | 0.261 +/- 0.026 | 0.261 +/- 0.026 | 0.907 +/- 0.014 | 0.914 +/- 0.010 |

### Key Findings (Synthetic)

- **ACGMT dominates on correlated and irrelevant** datasets, outperforming even GradientBoosting (R² 0.983 vs 0.978 and 0.992 vs 0.988). The coefficient-guided heuristic is a natural fit when the true structure is linear with a few predictive features.
- **RHM fails on interaction and regime** (R² 0.42 and 0.26) because top-1 feature selection commits to the wrong split axis. ACGMT's top-k preselection recovers performance (0.95 and 0.91).
- **GradientBoosting wins on scaling** (0.970 vs ACGMT's 0.854). The quadratic term in the DGP exceeds what a single-depth-level linear leaf can capture.
- **RHM and ACGMT (top1, no-std) produce identical results**, confirming that the ACGMT codebase faithfully reproduces RHM behavior when enhancements are disabled.

---

## Results: Real-World Datasets

### Test R² (mean +/- std over 5 seeds)

| Dataset | Ridge | CART | RandomForest | GradientBoosting | RHM | ACGMT (top1) | ACGMT (top3, std) | ACGMT (full) |
|---------|-------|------|--------------|------------------|-----|--------------|-------------------|--------------|
| california | 0.606 +/- 0.020 | 0.569 +/- 0.014 | 0.611 +/- 0.013 | **0.811 +/- 0.009** | -0.712 +/- 2.957 | -0.712 +/- 2.957 | 0.574 +/- 0.355 | 0.696 +/- 0.055 |
| diabetes | 0.412 +/- 0.034 | 0.265 +/- 0.099 | **0.428 +/- 0.076** | 0.352 +/- 0.093 | 0.371 +/- 0.059 | 0.371 +/- 0.059 | 0.357 +/- 0.075 | 0.383 +/- 0.030 |
| concrete | 0.621 +/- 0.022 | 0.670 +/- 0.075 | 0.782 +/- 0.038 | **0.923 +/- 0.019** | 0.734 +/- 0.027 | 0.734 +/- 0.027 | 0.844 +/- 0.013 | 0.838 +/- 0.011 |
| airfoil | 0.477 +/- 0.030 | 0.537 +/- 0.058 | 0.630 +/- 0.030 | **0.884 +/- 0.011** | 0.621 +/- 0.051 | 0.621 +/- 0.051 | 0.756 +/- 0.029 | 0.702 +/- 0.044 |
| yacht | 0.602 +/- 0.037 | 0.994 +/- 0.001 | 0.995 +/- 0.002 | **0.998 +/- 0.001** | 0.632 +/- 0.044 | 0.632 +/- 0.044 | 0.632 +/- 0.044 | 0.577 +/- 0.041 |
| friedman1 | 0.755 +/- 0.007 | 0.659 +/- 0.018 | 0.736 +/- 0.011 | **0.969 +/- 0.003** | 0.752 +/- 0.011 | 0.752 +/- 0.011 | 0.881 +/- 0.004 | 0.880 +/- 0.005 |

### Test MSE (mean +/- std over 5 seeds)

| Dataset | Ridge | CART | RandomForest | GradientBoosting | RHM | ACGMT (full) |
|---------|-------|------|--------------|------------------|-----|--------------|
| california | 0.527 +/- 0.019 | 0.577 +/- 0.011 | 0.520 +/- 0.012 | **0.253 +/- 0.008** | 2.280 +/- 3.931 | 0.406 +/- 0.069 |
| diabetes | 3381 +/- 283 | 4214 +/- 546 | **3277 +/- 404** | 3721 +/- 562 | 3606 +/- 245 | 3551 +/- 335 |
| concrete | 101.1 +/- 6.1 | 87.6 +/- 16.9 | 57.9 +/- 8.2 | **20.5 +/- 4.5** | 70.8 +/- 4.7 | 43.3 +/- 1.6 |
| airfoil | 25.0 +/- 1.3 | 22.0 +/- 2.2 | 17.7 +/- 1.1 | **5.5 +/- 0.5** | 18.0 +/- 1.6 | 14.2 +/- 1.6 |
| yacht | 82.8 +/- 31.6 | 1.21 +/- 0.42 | 0.98 +/- 0.63 | **0.42 +/- 0.35** | 75.8 +/- 28.1 | 88.4 +/- 34.5 |
| friedman1 | 5.80 +/- 0.14 | 8.08 +/- 0.30 | 6.26 +/- 0.13 | **0.73 +/- 0.07** | 5.87 +/- 0.23 | 2.85 +/- 0.10 |

### Test MAE (mean over 5 seeds)

| Dataset | Ridge | CART | RandomForest | GradientBoosting | RHM | ACGMT (full) |
|---------|-------|------|--------------|------------------|-----|--------------|
| california | 0.531 | 0.557 | 0.530 | **0.345** | 0.501 | 0.450 |
| diabetes | 48.6 | 51.2 | **46.7** | 49.2 | 48.9 | 50.1 |
| concrete | 8.03 | 7.20 | 6.03 | **3.25** | 6.66 | 4.94 |
| airfoil | 3.97 | 3.66 | 3.31 | **1.71** | 3.34 | 2.84 |
| yacht | 6.45 | 0.68 | 0.55 | **0.28** | 4.99 | 6.57 |
| friedman1 | 1.87 | 2.27 | 1.99 | **0.67** | 1.88 | 1.39 |

---

## Overfitting Analysis

The train-test R² gap measures how much the model overfits to training data. A small gap indicates good generalization.

### Train-Test R² Gap (mean over 5 seeds)

| Dataset | Ridge | CART | RandomForest | GradientBoosting | RHM | ACGMT (full) |
|---------|-------|------|--------------|------------------|-----|--------------|
| california | 0.000 | 0.018 | 0.015 | 0.029 | **1.382** | 0.001 |
| diabetes | 0.027 | **0.327** | 0.235 | **0.584** | 0.068 | 0.038 |
| concrete | -0.008 | 0.062 | 0.038 | **0.050** | 0.037 | 0.027 |
| airfoil | 0.010 | **0.064** | 0.055 | 0.054 | 0.019 | 0.015 |
| yacht | 0.001 | 0.002 | 0.002 | 0.002 | -0.012 | 0.006 |
| friedman1 | -0.010 | 0.030 | 0.023 | 0.015 | 0.010 | 0.008 |

**Notable findings:**

- **RHM on California Housing has a gap of 1.38**, indicating catastrophic overfitting. The model fits training data at R² 0.67 but produces negative R² on test data. This is caused by the top-1 coefficient heuristic selecting the wrong feature, leading to degenerate splits.
- **ACGMT (full) has consistently small gaps** (0.001-0.038), demonstrating that validation-aware pruning effectively controls overfitting.
- **GradientBoosting overfits on diabetes** (gap 0.58), achieving train R² 0.94 but test R² only 0.35. With only 353 training samples, 100 boosting rounds is excessive.
- **Ridge has near-zero gaps** everywhere, as expected from a linear model with L2 regularization.

---

## Timing

### Mean Fit Time (seconds, 5 seeds)

| Dataset | Ridge | CART | RandomForest | GradientBoosting | RHM | ACGMT (full) |
|---------|-------|------|--------------|------------------|-----|--------------|
| california | 0.006 | 0.111 | 10.150 | 16.352 | 3.695 | 4.621 |
| diabetes | 0.001 | 0.002 | 0.230 | 0.191 | 0.085 | 0.089 |
| concrete | 0.001 | 0.003 | 0.326 | 0.319 | 0.303 | 0.400 |
| airfoil | 0.002 | 0.003 | 0.259 | 0.165 | 0.439 | 0.581 |
| yacht | 0.002 | 0.002 | 0.164 | 0.098 | 0.044 | 0.003 |
| friedman1 | 0.003 | 0.042 | 2.205 | 3.283 | 1.016 | 2.278 |

- **Ridge** is the fastest model everywhere (< 0.01s).
- **CART** is second-fastest (< 0.12s).
- **ACGMT (full)** is comparable to or faster than GradientBoosting on most datasets, and ~3.5x faster on California Housing (4.6s vs 16.4s).
- **RHM and ACGMT (top1)** are faster than full ACGMT because they evaluate only 1 candidate feature per node instead of 3.

---

## Per-Dataset Analysis

### California Housing (n=20,640, d=8)

The largest dataset. Features include median income, house age, average rooms, population, and geographic coordinates (latitude, longitude).

| Model | Test R² | Test MAE | Fit Time |
|-------|---------|----------|----------|
| GradientBoosting | **0.811** | **0.345** | 16.4s |
| ACGMT (full) | 0.696 | 0.450 | 4.6s |
| RandomForest | 0.611 | 0.530 | 10.2s |
| Ridge | 0.606 | 0.531 | 0.006s |
| ACGMT (top3, std) | 0.574 | 0.404 | 3.2s |
| CART | 0.569 | 0.557 | 0.1s |
| RHM (original) | -0.712 | 0.501 | 3.7s |

**Analysis:** This dataset has strong nonlinear spatial structure (latitude/longitude interact with income to determine price). GradientBoosting captures this with 100 additive trees. ACGMT (full) reaches R² 0.696 — a substantial improvement over Ridge (0.606) — by discovering meaningful piecewise regions, but cannot match an ensemble of 100 trees. RHM fails catastrophically (negative R²) because the top-1 coefficient heuristic picks the wrong feature on some seeds, leading to degenerate tree structures. ACGMT's top-k preselection and standardization prevent this failure mode, with a very high variance in RHM (std = 2.96) compared to ACGMT full (std = 0.055). ACGMT (full) is also ~3.5x faster than GradientBoosting (4.6s vs 16.4s).

### Diabetes (n=442, d=10)

The smallest dataset. All features are pre-normalized. Target is disease progression.

| Model | Test R² | Test MAE | Fit Time |
|-------|---------|----------|----------|
| RandomForest | **0.428** | **46.7** | 0.23s |
| Ridge | 0.412 | 48.6 | 0.001s |
| ACGMT (full) | 0.383 | 50.1 | 0.089s |
| RHM (original) | 0.371 | 48.9 | 0.085s |
| ACGMT (top3, std) | 0.357 | 50.2 | 0.074s |
| GradientBoosting | 0.352 | 49.2 | 0.19s |
| CART | 0.265 | 51.2 | 0.002s |

**Analysis:** With only 353 training samples, all models struggle. No model exceeds R² 0.43. Ridge performs nearly as well as the best model (RandomForest), suggesting the underlying relationship is close to linear. Tree-based models (including ACGMT) lack enough data to discover reliable piecewise structure — splits on 353 samples produce leaves too small for stable Ridge fits. GradientBoosting massively overfits (train R² 0.94, test R² 0.35).

### Concrete Compressive Strength (n=1,030, d=8)

Cement composition and age predict compressive strength. Known nonlinear effects (strength increases nonlinearly with age and depends on component interactions).

| Model | Test R² | Test MAE | Fit Time |
|-------|---------|----------|----------|
| GradientBoosting | **0.923** | **3.25** | 0.32s |
| ACGMT (top3, std) | 0.844 | 4.75 | 0.33s |
| ACGMT (full) | 0.838 | 4.94 | 0.40s |
| RandomForest | 0.782 | 6.03 | 0.33s |
| RHM (original) | 0.734 | 6.66 | 0.30s |
| CART | 0.670 | 7.20 | 0.003s |
| Ridge | 0.621 | 8.03 | 0.001s |

**Analysis:** ACGMT shows strong improvement over RHM (0.838 vs 0.734, +14 percentage points). Top-k preselection and standardization are the drivers — note that ACGMT (top3, std) at 0.844 slightly outperforms ACGMT (full) at 0.838, suggesting that pruning removes some useful splits on this moderate-sized dataset. Both ACGMT variants substantially outperform Ridge (+22pp) and CART (+17pp), confirming that piecewise linear models capture the nonlinear cement-age interactions. ACGMT (full) also has notably low variance (std = 0.011) compared to CART (std = 0.075).

### Airfoil Self-Noise (n=1,503, d=5)

NASA dataset measuring noise from airfoil sections. Five aeroacoustic features predict sound pressure level.

| Model | Test R² | Test MAE | Fit Time |
|-------|---------|----------|----------|
| GradientBoosting | **0.884** | **1.71** | 0.17s |
| ACGMT (top3, std) | 0.756 | 2.54 | 0.41s |
| ACGMT (full) | 0.702 | 2.84 | 0.58s |
| RandomForest | 0.630 | 3.31 | 0.26s |
| RHM (original) | 0.621 | 3.34 | 0.44s |
| CART | 0.537 | 3.66 | 0.003s |
| Ridge | 0.477 | 3.97 | 0.002s |

**Analysis:** ACGMT (top3, std) provides the best single-tree result at R² 0.756 — a +14pp improvement over RHM. Interestingly, ACGMT (full) at 0.702 is lower than the unpruned ACGMT (top3, std), indicating that pruning is too aggressive on this 1,503-sample dataset. The validation holdout (20%) reduces the effective training set to ~960 samples, limiting the tree's ability to fit fine-grained piecewise structure. Ridge at 0.477 confirms that the relationship is far from globally linear — piecewise models capture meaningful regime structure in the aeroacoustic data.

### Yacht Hydrodynamics (n=308, d=6)

Very small dataset. Residuary resistance depends heavily on the Froude number with a highly nonlinear (exponential-like) relationship.

| Model | Test R² | Test MAE | Fit Time |
|-------|---------|----------|----------|
| GradientBoosting | **0.998** | **0.28** | 0.098s |
| RandomForest | 0.995 | 0.55 | 0.16s |
| CART | 0.994 | 0.68 | 0.002s |
| RHM (original) | 0.632 | 4.99 | 0.044s |
| ACGMT (top3, std) | 0.632 | 4.99 | 0.027s |
| Ridge | 0.602 | 6.45 | 0.002s |
| ACGMT (full) | 0.577 | 6.57 | 0.003s |

**Analysis:** The yacht dataset is an adversarial case for linear-leaf model trees. The target has a near-exponential relationship with the Froude number, meaning that within each tree leaf, a linear model is a poor local approximation. CART, RandomForest, and GradientBoosting (which use constant-leaf predictions) achieve near-perfect R² because they can place many splits to approximate the curve. RHM and ACGMT (all variants) cap out around R² 0.63 — their Ridge leaf models cannot capture the curvature within each partition. ACGMT (full) actually performs worst (0.577) because pruning collapses useful splits given only 246 training samples.

### Friedman #1 (n=5,000, d=10)

Standard synthetic benchmark: y = 10*sin(pi*x1*x2) + 20*(x3-0.5)^2 + 10*x4 + 5*x5 + noise. Five informative features, five noise features.

| Model | Test R² | Test MAE | Fit Time |
|-------|---------|----------|----------|
| GradientBoosting | **0.969** | **0.67** | 3.3s |
| ACGMT (top3, std) | 0.881 | 1.39 | 1.6s |
| ACGMT (full) | 0.880 | 1.39 | 2.3s |
| Ridge | 0.755 | 1.87 | 0.003s |
| RHM (original) | 0.752 | 1.88 | 1.0s |
| RandomForest | 0.736 | 1.99 | 2.2s |
| CART | 0.659 | 2.27 | 0.04s |

**Analysis:** ACGMT achieves R² 0.88, a +13pp jump over RHM (0.75) and comparable to Ridge despite the strong nonlinearity in the DGP. The sinusoidal interaction term sin(pi*x1*x2) creates natural piecewise regimes that ACGMT discovers through coefficient-guided splitting. Interestingly, RHM performs at nearly the same level as Ridge (0.752 vs 0.755), suggesting that without top-k and standardization, the coefficient heuristic fails to find useful splits on this problem. GradientBoosting leads at 0.969 by additively refining the residuals over 100 rounds.

---

## Cross-Cutting Findings

### 1. ACGMT consistently improves over RHM on real data

Across all 6 real-world/standard datasets, ACGMT (full) matches or outperforms RHM:

| Dataset | RHM R² | ACGMT (full) R² | Improvement |
|---------|--------|-----------------|-------------|
| california | -0.712 | 0.696 | +1.408 |
| friedman1 | 0.752 | 0.880 | +0.128 |
| concrete | 0.734 | 0.838 | +0.104 |
| airfoil | 0.621 | 0.702 | +0.081 |
| diabetes | 0.371 | 0.383 | +0.012 |
| yacht | 0.632 | 0.577 | -0.055 |

The sole exception is yacht (308 samples, highly nonlinear), where both methods struggle equally and pruning slightly hurts.

### 2. Top-k preselection is the dominant enhancement

Comparing RHM (top-1) to ACGMT (top3, std) isolates the combined effect of top-k and standardization. The improvement is dramatic on interaction (+0.53), regime (+0.65), friedman1 (+0.13), and concrete (+0.11). On datasets where top-1 already picks the right feature (correlated, irrelevant), the gains are marginal.

### 3. Pruning helps on large datasets, hurts on small ones

On datasets with sufficient data (california, regime, interaction), pruning reduces overfitting and improves or maintains test R². On smaller datasets (yacht, airfoil, concrete), pruning collapses useful splits because the 20% validation holdout leaves too few samples for reliable pruning decisions. ACGMT (top3, std) without pruning outperforms ACGMT (full) on airfoil (0.756 vs 0.702) and concrete (0.844 vs 0.838).

### 4. GradientBoosting dominates on real-world data

GradientBoosting achieves the best test R² on 5 of 6 real-world datasets (all except diabetes). This is expected — an ensemble of 100 additive trees with depth 4 has far more capacity than a single depth-4 tree with linear leaves. The gap is largest on highly nonlinear datasets (yacht: 0.998 vs 0.577, california: 0.811 vs 0.696) and smallest where the true structure is piecewise linear.

### 5. Linear-leaf trees struggle with exponential/highly nonlinear targets

The yacht and scaling datasets expose a fundamental limitation: when the target-feature relationship is exponential or strongly nonlinear within each leaf partition, a local Ridge model is a poor approximation. Constant-leaf trees (CART, RF, GBR) handle this better by simply averaging within each partition. This limitation is inherent to the linear model tree family, not specific to RHM/ACGMT.

### 6. ACGMT provides a favorable interpretability-accuracy tradeoff

While GradientBoosting outperforms ACGMT on raw accuracy, ACGMT produces a single interpretable tree with depth <= 4 (at most 16 leaves), each containing a named linear equation. On datasets where the true structure is piecewise linear (correlated, irrelevant, interaction, concrete), ACGMT approaches or matches GradientBoosting's accuracy while remaining fully inspectable.

---

## Reproducibility

All experiments can be reproduced with:

```bash
pip install numpy pandas scikit-learn matplotlib
python experiments.py      # Runs all 11 datasets x 5 seeds x 8 models (parallelized via joblib)
python ablations.py        # Ablation studies on synthetic datasets
python generate_summary.py # Aggregate results
```

Results are saved to `results/comparison_full.csv` (440 rows) and `results/comparison_summary.csv` (88 rows, aggregated across seeds).

Hardware: experiments parallelized with joblib using 8 concurrent workers. Total wall time approximately 10 minutes.
