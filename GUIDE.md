# User Guide

## Installation

Clone the repository and install dependencies:

```bash
git clone git@github.com:model-tree-acgmt/model-tree-acgmt.git
cd model-tree-acgmt
pip install numpy pandas scikit-learn matplotlib
```

No package installation is needed — import directly from the source files.

---

## Part 1: RHM — Recursive Hybrid Model

RHM is the base algorithm (Paper 1). It's minimal by design: a binary tree where each internal node picks a split feature using Ridge coefficient magnitude, and each leaf holds a local Ridge model.

### How RHM works

1. **Fit Ridge** on the current node's data
2. **Pick the feature** with the largest absolute coefficient
3. **Search a 9-point percentile grid** (10th–90th) on that feature for the split that minimizes weighted child MSE
4. **Split and recurse** until max depth or minimum sample size is reached
5. **Fit a Ridge leaf model** at each terminal node

### Using RHM

```python
from code import RecursiveHybridModel

model = RecursiveHybridModel(max_depth=3, min_samples_split=100)
model.fit(X_train, y_train)
predictions = model.predict(X_test)
```

**Parameters:**
- `max_depth` (default 2) — maximum tree depth
- `min_samples_split` (default 100) — minimum samples to attempt a split

RHM accepts both NumPy arrays and pandas DataFrames.

### Visualizing the tree

RHM can generate Graphviz DOT files for tree visualization:

```bash
python code.py --auto_generate_png
```

This generates `hybrid_tree_depthN.dot` and `.png` files (requires the `dot` command from Graphviz).

### RHM limitations

RHM commits to a single feature per node (top-1). This breaks down when:
- Features are on **different scales** — the largest coefficient doesn't mean the most informative feature
- Features are **correlated** — signal gets spread across correlated variables
- The data has **regime structure** where the best split feature isn't the one with the strongest global linear signal

These limitations motivate ACGMT.

---

## Part 2: ACGMT — Adaptive Coefficient-Guided Model Tree

ACGMT (Paper 2) keeps RHM's structure but fixes its split search. The key change is **top-k feature preselection**: instead of committing to one feature, ACGMT screens the k best candidates and searches each for the best split.

### How ACGMT works

1. **Standardize features** (optional) at the current node, then **fit Ridge**
2. **Select top-k features** by absolute coefficient magnitude
3. **For each candidate feature**, search for the best split point:
   - Coarse pass: 9 quantile points (10th–90th percentile)
   - Fine pass (if enabled): refine around the best coarse point
   - At each candidate, fit Ridge on both sides and compute weighted MSE
4. **Split on the best (feature, threshold)** across all k candidates and recurse
5. **Stop** at max depth or minimum samples — fit a Ridge leaf
6. **Prune** (if enabled): bottom-up, collapse subtrees where a single leaf does as well on validation data

### Using ACGMT

```python
from acgmt import AdaptiveCoefficientGuidedModelTree

model = AdaptiveCoefficientGuidedModelTree(max_depth=4, min_samples_split=100)
model.fit(X_train, y_train)
predictions = model.predict(X_test)
```

### Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `max_depth` | 4 | Maximum tree depth |
| `min_samples_split` | 100 | Minimum samples to attempt a split |
| `alpha` | 1.0 | Ridge regularization strength |
| `top_k` | 3 | Number of top features to evaluate as split candidates |
| `standardize` | True | Standardize features per node before ranking coefficients |
| `adaptive_grid` | True | Coarse-to-fine split search |
| `coarse_quantiles` | 9 | Quantile points in the coarse pass |
| `fine_quantiles` | 5 | Points in the fine refinement pass |
| `fine_window` | 0.1 | Width of the fine search window (fraction of coarse range) |
| `val_fraction` | 0.2 | Fraction of training data held out for pruning (0 to disable) |
| `prune` | True | Validation-aware bottom-up pruning |
| `random_state` | 42 | Random seed for reproducibility |

### Parameter tuning

**Start simple, then add complexity:**

```python
# Minimal (equivalent to RHM)
model = AdaptiveCoefficientGuidedModelTree(
    max_depth=3, top_k=1, standardize=False,
    adaptive_grid=False, prune=False, val_fraction=0,
)

# Recommended defaults (all enhancements)
model = AdaptiveCoefficientGuidedModelTree(
    max_depth=4, min_samples_split=100, top_k=3,
    standardize=True, adaptive_grid=True,
    prune=True, val_fraction=0.2,
)
```

- **`top_k`**: 3 is a good default. Increase if features are correlated. Going beyond 5 rarely helps.
- **`max_depth`**: increase if you expect many regimes. If train R² >> test R², reduce depth or enable pruning.
- **`min_samples_split`**: increase for small datasets, decrease for large ones. Rule of thumb: 20–30 samples per feature at each leaf.
- **`standardize`**: leave on unless all features are already on the same scale.

### scikit-learn compatibility

ACGMT supports `get_params()` and `set_params()`:

```python
from sklearn.model_selection import cross_val_score, GridSearchCV

# Cross-validation
scores = cross_val_score(
    AdaptiveCoefficientGuidedModelTree(), X, y, cv=5, scoring='r2'
)

# Grid search
param_grid = {'max_depth': [2, 3, 4], 'top_k': [1, 3, 5], 'prune': [True, False]}
search = GridSearchCV(
    AdaptiveCoefficientGuidedModelTree(),
    param_grid, cv=3, scoring='neg_mean_squared_error'
)
search.fit(X, y)
print(search.best_params_)
```

---

## Using on your own data

### Complete workflow

```python
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from acgmt import AdaptiveCoefficientGuidedModelTree

# Load your data
df = pd.read_csv('your_data.csv')

# Separate features and target
target_column = 'price'  # change to your target column
X = df.drop(columns=[target_column])
y = df[target_column]

# Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Fit
model = AdaptiveCoefficientGuidedModelTree(max_depth=4, min_samples_split=100)
model.fit(X_train, y_train)

# Evaluate
train_pred = model.predict(X_train)
test_pred = model.predict(X_test)
print(f"Train R²: {r2_score(y_train, train_pred):.4f}")
print(f"Test  R²: {r2_score(y_test, test_pred):.4f}")
print(f"Test MSE: {mean_squared_error(y_test, test_pred):.4f}")
```

### Data requirements

Both RHM and ACGMT fit Ridge regression at every node, so standard linear model preprocessing applies:

- **Numeric features only.** Encode categoricals first (one-hot or ordinal). No built-in categorical handling.
- **No missing values.** Impute or drop NaNs before fitting.
- **Feature scaling is handled internally** when `standardize=True` (ACGMT default). You don't need to scale features yourself. For RHM, consider scaling manually since it has no built-in standardization.
- **Outliers matter.** Ridge regression is sensitive to extremes. Consider clipping or log-transforming heavy-tailed features.

```python
# Example: preprocessing a real dataset
from sklearn.preprocessing import OrdinalEncoder
import numpy as np

cat_cols = df.select_dtypes(include='object').columns.tolist()
num_cols = df.select_dtypes(include='number').columns.drop(target_column).tolist()

if cat_cols:
    encoder = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
    df[cat_cols] = encoder.fit_transform(df[cat_cols])

df = df.dropna()

# Optional: log-transform skewed features
for col in num_cols:
    if df[col].skew() > 2 and (df[col] > 0).all():
        df[col] = np.log1p(df[col])

X = df[num_cols + cat_cols]
y = df[target_column]
```

### When to use these models

**Good fit:**
- Locally linear relationships — different linear models apply in different regions (regime-dependent behavior)
- Interpretability matters — you can inspect tree splits and Ridge coefficients at each leaf
- Moderate number of features (up to ~50)
- Enough data to populate leaves (at least ~1,600 samples for `max_depth=4, min_samples_split=100`)

**Not the best choice:**
- Highly nonlinear with no natural regimes — use Gradient Boosting or Random Forest
- Very high-dimensional (hundreds of features) — per-node Ridge fits become expensive
- Very few samples (<500) — not enough to fill meaningful leaf models

### Inspecting the fitted tree

Walk the tree to see what the model learned:

```python
def print_tree(node, feature_names, depth=0):
    indent = "  " * depth
    if node.model is not None:
        coefs = dict(zip(feature_names, node.model.coef_))
        top = sorted(coefs.items(), key=lambda x: abs(x[1]), reverse=True)[:5]
        coef_str = ", ".join(f"{k}: {v:+.3f}" for k, v in top)
        print(f"{indent}LEAF (n={node.n_samples}) intercept={node.model.intercept_:.3f} [{coef_str}]")
    else:
        fname = feature_names[node.feature] if isinstance(node.feature, int) else node.feature
        print(f"{indent}SPLIT {fname} <= {node.split_point:.3f}")
        print_tree(node.left, feature_names, depth + 1)
        print_tree(node.right, feature_names, depth + 1)

# For ACGMT
print_tree(model.root, model.feature_names_)
```

For RHM, the tree structure is the same but `feature` stores a string name (not an index) and nodes don't have `n_samples`:

```python
def print_rhm_tree(node, depth=0):
    indent = "  " * depth
    if node.model is not None:
        coef_str = ", ".join(f"{c:.3f}" for c in node.model.coef_[:5])
        print(f"{indent}LEAF intercept={node.model.intercept_:.3f} coefs=[{coef_str}]")
    else:
        print(f"{indent}SPLIT {node.feature} <= {node.split_point:.3f}")
        print_rhm_tree(node.left, depth + 1)
        print_rhm_tree(node.right, depth + 1)

print_rhm_tree(model.root)
```

---

## Running experiments

### Full comparison (all models x all datasets)

```bash
python experiments.py
```

Outputs: `results/comparison_full.csv`, `results/comparison_summary.csv`

### Ablation studies

```bash
python ablations.py
```

Four ablation axes:
- **top-k**: k = 1, 2, 3, 5
- **standardization**: with vs without
- **adaptive grid**: fixed vs adaptive
- **pruning**: with vs without (both on same 80% build set)

Outputs: `results/ablation_*_full.csv`, `results/ablation_*_summary.csv`

### Generate summary tables

```bash
python generate_summary.py
```

Outputs: `results/summary.txt`

---

## Synthetic datasets

Five datasets in `datasets.py`, each targeting a specific failure mode:

| Dataset | What it tests |
|---------|--------------|
| `scaling` | Features at different scales (1x to 1000x) |
| `correlated` | Correlated feature pairs (rho=0.95) |
| `irrelevant` | 3 relevant among 17 noise features |
| `interaction` | Regime shift at x0=0 with interaction effects |
| `regime` | Piecewise linear with 4 breakpoints |

```python
from datasets import ALL_DATASETS

(X_train, X_test, y_train, y_test), feature_names = ALL_DATASETS['interaction'](random_state=42)
```
