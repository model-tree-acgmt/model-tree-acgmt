"""Adaptive Coefficient-Guided Model Tree (ACGMT).

Enhancements over the original RHM:
1. Node-wise feature standardization for scale-invariant coefficient ranking
2. Top-k feature preselection instead of top-1
3. Coarse-to-fine adaptive threshold search
4. Validation-aware bottom-up pruning
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split


class ACGMTNode:
    __slots__ = ('feature', 'split_point', 'left', 'right', 'model',
                 'n_samples', 'build_indices')

    def __init__(self, feature=None, split_point=None, left=None, right=None,
                 model=None, n_samples=0, build_indices=None):
        self.feature = feature
        self.split_point = split_point
        self.left = left
        self.right = right
        self.model = model
        self.n_samples = n_samples
        self.build_indices = build_indices


class AdaptiveCoefficientGuidedModelTree:
    """Piecewise linear regression tree with adaptive coefficient-guided splitting."""

    def __init__(self, max_depth=4, min_samples_split=100, alpha=1.0,
                 top_k=3, standardize=True,
                 adaptive_grid=True, coarse_quantiles=9,
                 fine_quantiles=5, fine_window=0.1,
                 val_fraction=0.2, prune=True, random_state=42):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.alpha = alpha
        self.top_k = top_k
        self.standardize = standardize
        self.adaptive_grid = adaptive_grid
        self.coarse_quantiles = coarse_quantiles
        self.fine_quantiles = fine_quantiles
        self.fine_window = fine_window
        self.val_fraction = val_fraction
        self.prune = prune
        self.random_state = random_state
        self.root = None
        self.feature_names_ = None
        self.n_features_ = None

    def get_params(self, deep=True):
        return {k: getattr(self, k) for k in [
            'max_depth', 'min_samples_split', 'alpha', 'top_k', 'standardize',
            'adaptive_grid', 'coarse_quantiles', 'fine_quantiles', 'fine_window',
            'val_fraction', 'prune', 'random_state',
        ]}

    def set_params(self, **params):
        for k, v in params.items():
            setattr(self, k, v)
        return self

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit(self, X, y):
        X, y = self._validate_input(X, y)
        self.n_features_ = X.shape[1]

        # Always split when val_fraction > 0 so both pruned and unpruned
        # conditions build on the same data subset (clean ablation).
        if self.val_fraction > 0:
            X_build, X_val, y_build, y_val, _, _ = train_test_split(
                X, y, np.arange(len(y)),
                test_size=self.val_fraction, random_state=self.random_state)
        else:
            X_build, y_build = X, y
            X_val, y_val = None, None

        self.root = self._build_tree(X_build, y_build, np.arange(len(y_build)), depth=0)

        if self.prune and X_val is not None:
            self._prune_tree(self.root, X_build, y_build, X_val, y_val)

        # Clear build indices to free memory
        self._clear_indices(self.root)
        return self

    def predict(self, X):
        X, _ = self._validate_input(X, None)
        predictions = np.empty(len(X))
        self._predict_bulk(self.root, X, np.arange(len(X)), predictions)
        return predictions

    # ------------------------------------------------------------------
    # Tree construction
    # ------------------------------------------------------------------

    def _build_tree(self, X, y, indices, depth):
        n = len(y)
        node = ACGMTNode(n_samples=n, build_indices=indices)

        # Leaf conditions
        if depth >= self.max_depth or n < self.min_samples_split:
            node.model = Ridge(alpha=self.alpha).fit(X, y)
            return node

        # Enhancement 1: standardize for coefficient ranking
        if self.standardize:
            mu = X.mean(axis=0)
            sigma = X.std(axis=0)
            sigma[sigma < 1e-8] = 1.0
            X_std = (X - mu) / sigma
        else:
            X_std = X

        ridge = Ridge(alpha=self.alpha).fit(X_std, y)
        coef_abs = np.abs(ridge.coef_)

        # Enhancement 2: top-k feature preselection
        k = min(self.top_k, X.shape[1])
        top_features = np.argsort(coef_abs)[-k:]

        # Search best split across top-k features
        best_mse = np.inf
        best_feature = None
        best_split = None
        best_left_mask = None

        for feat_idx in top_features:
            col = X[:, feat_idx]
            split, mse, left_mask = self._find_best_split(X, y, col, feat_idx)
            if split is not None and mse < best_mse:
                best_mse = mse
                best_feature = feat_idx
                best_split = split
                best_left_mask = left_mask

        # Fallback to leaf if no valid split
        if best_split is None:
            node.model = Ridge(alpha=self.alpha).fit(X, y)
            return node

        right_mask = ~best_left_mask
        node.feature = best_feature
        node.split_point = best_split
        node.left = self._build_tree(
            X[best_left_mask], y[best_left_mask],
            indices[best_left_mask], depth + 1)
        node.right = self._build_tree(
            X[right_mask], y[right_mask],
            indices[right_mask], depth + 1)
        return node

    def _find_best_split(self, X, y, col, feat_idx):
        """Find the best split point for a single feature column."""
        n = len(y)
        # Coarse pass
        pcts = np.linspace(10, 90, self.coarse_quantiles)
        candidates = np.percentile(col, pcts)
        candidates = np.unique(candidates)

        best_mse = np.inf
        best_split = None
        best_mask = None

        for tau in candidates:
            mse, mask = self._eval_split(X, y, col, tau, n)
            if mse is not None and mse < best_mse:
                best_mse = mse
                best_split = tau
                best_mask = mask

        # Enhancement 3: adaptive fine pass
        if self.adaptive_grid and best_split is not None and len(candidates) >= 2:
            span = candidates[-1] - candidates[0]
            half_w = self.fine_window * span
            lo = max(best_split - half_w, np.percentile(col, 5))
            hi = min(best_split + half_w, np.percentile(col, 95))
            if hi > lo:
                fine_candidates = np.linspace(lo, hi, self.fine_quantiles + 2)[1:-1]
                for tau in fine_candidates:
                    mse, mask = self._eval_split(X, y, col, tau, n)
                    if mse is not None and mse < best_mse:
                        best_mse = mse
                        best_split = tau
                        best_mask = mask

        return best_split, best_mse, best_mask

    def _eval_split(self, X, y, col, tau, n):
        left_mask = col <= tau
        n_left = left_mask.sum()
        n_right = n - n_left
        if n_left < self.min_samples_split or n_right < self.min_samples_split:
            return None, None
        right_mask = ~left_mask
        m_l = Ridge(alpha=self.alpha).fit(X[left_mask], y[left_mask])
        m_r = Ridge(alpha=self.alpha).fit(X[right_mask], y[right_mask])
        mse = (mean_squared_error(y[left_mask], m_l.predict(X[left_mask])) * n_left +
               mean_squared_error(y[right_mask], m_r.predict(X[right_mask])) * n_right) / n
        return mse, left_mask

    # ------------------------------------------------------------------
    # Enhancement 4: validation-aware pruning
    # ------------------------------------------------------------------

    def _prune_tree(self, node, X_build, y_build, X_val, y_val):
        """Bottom-up pruning using validation set.

        Routes validation indices top-down so each node only sees the
        validation samples that actually reach it.
        """
        self._prune_node(node, X_build, y_build,
                         X_val, y_val, np.arange(len(X_val)))

    def _prune_node(self, node, X_build, y_build, X_val, y_val, val_idx):
        """Prune a single node given the validation indices that reach it."""
        if node.model is not None:
            return  # already a leaf

        # Route validation samples to children
        col = X_val[val_idx, node.feature]
        left_mask = col <= node.split_point
        left_val_idx = val_idx[left_mask]
        right_val_idx = val_idx[~left_mask]

        # Recurse into children first (bottom-up)
        self._prune_node(node.left, X_build, y_build,
                         X_val, y_val, left_val_idx)
        self._prune_node(node.right, X_build, y_build,
                         X_val, y_val, right_val_idx)

        # Need enough validation samples at this node to evaluate
        if len(val_idx) < 2:
            return

        # Subtree predictions on the validation samples that reach this node
        full_preds = np.empty(len(X_val))
        self._predict_bulk(node, X_val, val_idx, full_preds)
        subtree_preds = full_preds[val_idx]
        subtree_mse = mean_squared_error(y_val[val_idx], subtree_preds)

        # Fit a leaf on the build data that reached this node
        build_idx = self._collect_build_indices(node)
        if len(build_idx) < 2:
            return
        leaf_model = Ridge(alpha=self.alpha).fit(X_build[build_idx], y_build[build_idx])
        leaf_preds = leaf_model.predict(X_val[val_idx])
        leaf_mse = mean_squared_error(y_val[val_idx], leaf_preds)

        # Collapse if leaf is at least as good
        if leaf_mse <= subtree_mse:
            node.model = leaf_model
            node.left = None
            node.right = None
            node.feature = None
            node.split_point = None

    def _collect_build_indices(self, node):
        if node.build_indices is not None:
            return node.build_indices
        indices = []
        if node.left is not None:
            indices.append(self._collect_build_indices(node.left))
        if node.right is not None:
            indices.append(self._collect_build_indices(node.right))
        return np.concatenate(indices) if indices else np.array([], dtype=int)

    def _clear_indices(self, node):
        if node is None:
            return
        node.build_indices = None
        self._clear_indices(node.left)
        self._clear_indices(node.right)

    # ------------------------------------------------------------------
    # Prediction (vectorized)
    # ------------------------------------------------------------------

    def _predict_bulk(self, node, X, idx, out):
        if len(idx) == 0:
            return
        if node.model is not None:
            out[idx] = node.model.predict(X[idx])
            return
        col = X[idx, node.feature]
        left_local = col <= node.split_point
        left_idx = idx[left_local]
        right_idx = idx[~left_local]
        self._predict_bulk(node.left, X, left_idx, out)
        self._predict_bulk(node.right, X, right_idx, out)

    # ------------------------------------------------------------------
    # Input handling
    # ------------------------------------------------------------------

    def _validate_input(self, X, y):
        if isinstance(X, pd.DataFrame):
            if self.feature_names_ is None:
                self.feature_names_ = list(X.columns)
            X = X.values
        else:
            X = np.asarray(X)
            if self.feature_names_ is None:
                self.feature_names_ = [f'x{i}' for i in range(X.shape[1])]
        if y is not None:
            y = np.asarray(y, dtype=float)
        return X, y


if __name__ == '__main__':
    # Quick smoke test
    rng = np.random.RandomState(42)
    X = rng.randn(500, 5)
    y = 3*X[:, 0] - 2*X[:, 1] + X[:, 0]**2 + 0.3*rng.randn(500)

    for cfg_name, kwargs in [
        ('full', dict(top_k=3, standardize=True, adaptive_grid=True, prune=True)),
        ('top1-only', dict(top_k=1, standardize=False, adaptive_grid=False, prune=False)),
        ('no-prune', dict(top_k=3, standardize=True, adaptive_grid=True, prune=False)),
    ]:
        m = AdaptiveCoefficientGuidedModelTree(max_depth=3, min_samples_split=30, **kwargs)
        m.fit(X, y)
        preds = m.predict(X)
        mse = mean_squared_error(y, preds)
        print(f'{cfg_name:12s}  train MSE={mse:.4f}')

    print('Smoke test passed.')
