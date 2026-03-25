import numpy as np
from sklearn.model_selection import train_test_split


def make_scaling_data(n_samples=5000, noise=0.5, random_state=42):
    """Features at wildly different scales; true signal is equal after rescaling."""
    rng = np.random.RandomState(random_state)
    scales = np.array([1.0, 100.0, 0.01, 1000.0, 0.001])
    d = len(scales)
    X_raw = rng.randn(n_samples, d)
    X = X_raw * scales  # features at different scales
    # true coefficients in original scale: 1/scale (so contribution is equal)
    y = (X_raw * 2.0).sum(axis=1) + 0.5 * X_raw[:, 0]**2 + noise * rng.randn(n_samples)
    names = [f'x{i}_scale{s}' for i, s in enumerate(scales)]
    return train_test_split(X, y, test_size=0.2, random_state=random_state), names


def make_correlated_data(n_samples=5000, rho=0.95, noise=0.5, random_state=42):
    """Pairs of correlated features; only one per pair is truly predictive."""
    rng = np.random.RandomState(random_state)
    z1 = rng.randn(n_samples)
    z2 = rng.randn(n_samples)
    z3 = rng.randn(n_samples)
    # x0 is predictive, x1 is correlated noise copy
    x0 = z1
    x1 = rho * z1 + np.sqrt(1 - rho**2) * rng.randn(n_samples)
    # x2 is predictive, x3 is correlated noise copy
    x2 = z2
    x3 = rho * z2 + np.sqrt(1 - rho**2) * rng.randn(n_samples)
    x4 = z3  # independent predictive feature
    X = np.column_stack([x0, x1, x2, x3, x4])
    y = 3*x0 - 2*x2 + x4 + 0.5*x0**2 + noise * rng.randn(n_samples)
    names = ['x0_pred', 'x1_corr', 'x2_pred', 'x3_corr', 'x4_pred']
    return train_test_split(X, y, test_size=0.2, random_state=random_state), names


def make_irrelevant_data(n_samples=5000, n_relevant=3, n_irrelevant=17, noise=0.5,
                         random_state=42):
    """Few relevant features buried among many noise features."""
    rng = np.random.RandomState(random_state)
    d = n_relevant + n_irrelevant
    X = rng.randn(n_samples, d)
    # only first n_relevant features matter
    y = (5*X[:, 0] - 3*X[:, 1] + 2*X[:, 2]
         + 0.5*X[:, 0]**2 + X[:, 0]*X[:, 1]
         + noise * rng.randn(n_samples))
    names = [f'rel_{i}' for i in range(n_relevant)] + [f'irr_{i}' for i in range(n_irrelevant)]
    return train_test_split(X, y, test_size=0.2, random_state=random_state), names


def make_interaction_data(n_samples=5000, noise=0.5, random_state=42):
    """Strong interaction with regime shift at x0=0."""
    rng = np.random.RandomState(random_state)
    X = rng.randn(n_samples, 5)
    # regime-dependent interaction
    y = np.where(
        X[:, 0] <= 0,
        -2*X[:, 0] + 3*X[:, 1] + X[:, 0]*X[:, 1],
        4*X[:, 0] - X[:, 1] + 2*X[:, 0]*X[:, 1],
    ) + noise * rng.randn(n_samples)
    names = [f'x{i}' for i in range(5)]
    return train_test_split(X, y, test_size=0.2, random_state=random_state), names


def make_regime_data(n_samples=5000, n_regimes=4, noise=0.5, random_state=42):
    """Piecewise linear in x0 with n_regimes breakpoints."""
    rng = np.random.RandomState(random_state)
    X = rng.randn(n_samples, 5)
    x0 = X[:, 0]
    breakpoints = np.linspace(-2, 2, n_regimes + 1)
    slopes = np.array([1.0, -2.0, 3.0, -1.5, 2.5])[:n_regimes]
    intercepts = np.array([0.0, 1.0, -1.0, 0.5, -0.5])[:n_regimes]
    y = np.zeros(n_samples)
    for i in range(n_regimes):
        mask = (x0 >= breakpoints[i]) & (x0 < breakpoints[i+1])
        y[mask] = slopes[i] * x0[mask] + intercepts[i] + X[mask, 1]
    # handle edges
    y[x0 < breakpoints[0]] = slopes[0] * x0[x0 < breakpoints[0]] + intercepts[0] + X[x0 < breakpoints[0], 1]
    y[x0 >= breakpoints[-1]] = slopes[-1] * x0[x0 >= breakpoints[-1]] + intercepts[-1] + X[x0 >= breakpoints[-1], 1]
    y += noise * rng.randn(n_samples)
    names = [f'x{i}' for i in range(5)]
    return train_test_split(X, y, test_size=0.2, random_state=random_state), names


ALL_DATASETS = {
    'scaling': make_scaling_data,
    'correlated': make_correlated_data,
    'irrelevant': make_irrelevant_data,
    'interaction': make_interaction_data,
    'regime': make_regime_data,
}
