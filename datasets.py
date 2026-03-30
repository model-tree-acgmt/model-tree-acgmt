import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.datasets import (fetch_california_housing, load_diabetes,
                               fetch_openml, make_friedman1)


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


# ---------------------------------------------------------------------------
# Real-world datasets
# ---------------------------------------------------------------------------

def make_california_housing(random_state=42, **kwargs):
    """California housing prices (20k samples, 8 features)."""
    data = fetch_california_housing()
    X, y = data.data, data.target
    names = list(data.feature_names)
    return train_test_split(X, y, test_size=0.2, random_state=random_state), names


def make_diabetes(random_state=42, **kwargs):
    """Diabetes progression (442 samples, 10 pre-normalized features)."""
    data = load_diabetes()
    X, y = data.data, data.target
    names = list(data.feature_names)
    return train_test_split(X, y, test_size=0.2, random_state=random_state), names


def make_concrete(random_state=42, **kwargs):
    """Concrete compressive strength (1030 samples, 8 features)."""
    data = fetch_openml(data_id=4353, as_frame=True, parser="auto")
    if data.target is not None:
        X = data.data.values.astype(float)
        y = data.target.values.astype(float)
        names = list(data.data.columns)
    else:
        # No declared target; last column is the target
        df = data.data
        X = df.iloc[:, :-1].values.astype(float)
        y = df.iloc[:, -1].values.astype(float)
        names = list(df.columns[:-1])
    return train_test_split(X, y, test_size=0.2, random_state=random_state), names


def make_airfoil(random_state=42, **kwargs):
    """Airfoil self-noise (1503 samples, 5 features)."""
    data = fetch_openml(name="airfoil_self_noise", version=1, as_frame=True, parser="auto")
    X = data.data.values.astype(float)
    y = data.target.values.astype(float)
    names = list(data.data.columns)
    return train_test_split(X, y, test_size=0.2, random_state=random_state), names


def make_yacht(random_state=42, **kwargs):
    """Yacht hydrodynamics (308 samples, 6 features)."""
    data = fetch_openml(name="yacht_hydrodynamics", version=1, as_frame=True, parser="auto")
    X = data.data.values.astype(float)
    y = data.target.values.astype(float)
    names = list(data.data.columns)
    return train_test_split(X, y, test_size=0.2, random_state=random_state), names


def make_friedman1_data(n_samples=5000, noise=0.5, random_state=42):
    """Friedman #1: y = 10*sin(pi*x1*x2) + 20*(x3-0.5)^2 + 10*x4 + 5*x5 + noise."""
    X, y = make_friedman1(n_samples=n_samples, n_features=10, noise=noise,
                          random_state=random_state)
    names = [f'x{i}' for i in range(X.shape[1])]
    return train_test_split(X, y, test_size=0.2, random_state=random_state), names


# ---------------------------------------------------------------------------

SYNTHETIC_DATASETS = {
    'scaling': make_scaling_data,
    'correlated': make_correlated_data,
    'irrelevant': make_irrelevant_data,
    'interaction': make_interaction_data,
    'regime': make_regime_data,
    'friedman1': make_friedman1_data,
}

REAL_DATASETS = {
    'california': make_california_housing,
    'diabetes': make_diabetes,
    'concrete': make_concrete,
    'airfoil': make_airfoil,
    'yacht': make_yacht,
}

ALL_DATASETS = {**SYNTHETIC_DATASETS, **REAL_DATASETS}
