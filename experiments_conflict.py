"""Main comparison experiments: ACGMT vs baselines across synthetic and real-world datasets.

Includes:
- Main comparison (practical defaults)
- GBR complexity sweep (n_estimators: 1..100)
- ACGMT depth sweep (max_depth: 1..5)
"""

import os
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor

from code import RecursiveHybridModel
from acgmt import AdaptiveCoefficientGuidedModelTree
from datasets import ALL_DATASETS
from utils import evaluate_model, time_fit_predict, results_to_dataframe, save_results


def make_models(seed=42):
    """Model factories for the main practical-defaults comparison."""
    return {
        'Ridge': lambda: Ridge(alpha=1.0),
        'CART': lambda: DecisionTreeRegressor(max_depth=4, random_state=seed),
        'RandomForest': lambda: RandomForestRegressor(
            n_estimators=100, max_depth=4, random_state=seed, n_jobs=1),
        'GradientBoosting': lambda: GradientBoostingRegressor(
            n_estimators=100, max_depth=4, random_state=seed),
        'RHM (original)': lambda: RecursiveHybridModel(
            max_depth=4, min_samples_split=100),
        'ACGMT (top1, no-std)': lambda: AdaptiveCoefficientGuidedModelTree(
            max_depth=4, min_samples_split=100, top_k=1, standardize=False,
            adaptive_grid=False, prune=False, val_fraction=0, random_state=seed),
        'ACGMT (top3, std)': lambda: AdaptiveCoefficientGuidedModelTree(
            max_depth=4, min_samples_split=100, top_k=3, standardize=True,
            adaptive_grid=False, prune=False, val_fraction=0, random_state=seed),
        'ACGMT (full)': lambda: AdaptiveCoefficientGuidedModelTree(
            max_depth=4, min_samples_split=100, top_k=3, standardize=True,
            adaptive_grid=True, prune=True, val_fraction=0.2, random_state=seed),
    }


def _run_single(model_factory, model_name, ds_name, seed,
                X_train, y_train, X_test, y_test, experiment='main'):
    """Fit, time, evaluate a single model. Returns a result dict or None on failure."""
    model = model_factory()
    try:
        timing = time_fit_predict(model, X_train, y_train, X_test)
    except Exception as e:
        print(f"  SKIP {model_name} on {ds_name} seed={seed}: {e}")
        return None

    model2 = model_factory()
    model2.fit(X_train, y_train)
    metrics = evaluate_model(model2, X_train, y_train, X_test, y_test)

    return {
        'experiment': experiment,
        'dataset': ds_name,
        'model': model_name,
        'seed': seed,
        **metrics,
        **timing,
    }


def run_main_comparison(seeds=(42, 123, 456, 789, 1024)):
    """Practical-defaults comparison: all models x all datasets."""
    results = []
    for ds_name, ds_func in ALL_DATASETS.items():
        for seed in seeds:
            (X_train, X_test, y_train, y_test), _ = ds_func(random_state=seed)
            for model_name, model_factory in make_models(seed=seed).items():
                row = _run_single(model_factory, model_name, ds_name, seed,
                                  X_train, y_train, X_test, y_test,
                                  experiment='main')
                if row:
                    results.append(row)
        print(f"  [main] '{ds_name}' done")
    return results


def run_gbr_complexity_sweep(seeds=(42, 123, 456, 789, 1024),
                             n_estimators_grid=(1, 2, 4, 8, 16, 32, 64, 100),
                             max_depth=4):
    """GBR complexity curve: vary n_estimators with fixed depth.

    Also includes ACGMT (full) at depth=4 as the reference point so the
    complexity curve can show where a single interpretable tree sits.
    """
    results = []
    for ds_name, ds_func in ALL_DATASETS.items():
        for seed in seeds:
            (X_train, X_test, y_train, y_test), _ = ds_func(random_state=seed)

            for n_est in n_estimators_grid:
                name = f'GBR(n={n_est})'
                factory = lambda _n=n_est, _s=seed: GradientBoostingRegressor(
                    n_estimators=_n, max_depth=max_depth, random_state=_s)
                row = _run_single(factory, name, ds_name, seed,
                                  X_train, y_train, X_test, y_test,
                                  experiment='gbr_sweep')
                if row:
                    results.append(row)

            # ACGMT reference point
            factory = lambda _s=seed: AdaptiveCoefficientGuidedModelTree(
                max_depth=4, min_samples_split=100, top_k=3, standardize=True,
                adaptive_grid=True, prune=True, val_fraction=0.2, random_state=_s)
            row = _run_single(factory, 'ACGMT (full, d=4)', ds_name, seed,
                              X_train, y_train, X_test, y_test,
                              experiment='gbr_sweep')
            if row:
                results.append(row)

        print(f"  [gbr_sweep] '{ds_name}' done")
    return results


def run_acgmt_depth_sweep(seeds=(42, 123, 456, 789, 1024),
                          depth_grid=(1, 2, 3, 4, 5)):
    """ACGMT depth curve: vary max_depth with top_k=3, standardize=True."""
    results = []
    for ds_name, ds_func in ALL_DATASETS.items():
        for seed in seeds:
            (X_train, X_test, y_train, y_test), _ = ds_func(random_state=seed)

            for depth in depth_grid:
                name = f'ACGMT (d={depth})'
                factory = lambda _d=depth, _s=seed: AdaptiveCoefficientGuidedModelTree(
                    max_depth=_d, min_samples_split=100, top_k=3, standardize=True,
                    adaptive_grid=True, prune=True, val_fraction=0.2, random_state=_s)
                row = _run_single(factory, name, ds_name, seed,
                                  X_train, y_train, X_test, y_test,
                                  experiment='acgmt_depth')
                if row:
                    results.append(row)

        print(f"  [acgmt_depth] '{ds_name}' done")
    return results


def run_experiments(seeds=(42, 123, 456, 789, 1024)):
    """Run all experiment suites and save results."""
    os.makedirs('results', exist_ok=True)

    print("=== Main comparison (practical defaults) ===")
    main_results = run_main_comparison(seeds)

    print("\n=== GBR complexity sweep ===")
    gbr_results = run_gbr_complexity_sweep(seeds)

    print("\n=== ACGMT depth sweep ===")
    depth_results = run_acgmt_depth_sweep(seeds)

    # Save all results
    df_main = results_to_dataframe(main_results)
    save_results(df_main, 'results/comparison_full.csv')

    df_gbr = results_to_dataframe(gbr_results)
    save_results(df_gbr, 'results/gbr_complexity_sweep.csv')

    df_depth = results_to_dataframe(depth_results)
    save_results(df_depth, 'results/acgmt_depth_sweep.csv')

    # Summary table for main comparison
    summary = df_main.groupby(['dataset', 'model']).agg(
        test_mse_mean=('test_mse', 'mean'),
        test_mse_std=('test_mse', 'std'),
        test_r2_mean=('test_r2', 'mean'),
        test_r2_std=('test_r2', 'std'),
        fit_time_mean=('fit_time_s', 'mean'),
    ).reset_index()
    save_results(summary, 'results/comparison_summary.csv')

    # Summary for GBR sweep
    gbr_summary = df_gbr.groupby(['dataset', 'model']).agg(
        test_r2_mean=('test_r2', 'mean'),
        test_r2_std=('test_r2', 'std'),
        test_mse_mean=('test_mse', 'mean'),
        fit_time_mean=('fit_time_s', 'mean'),
    ).reset_index()
    save_results(gbr_summary, 'results/gbr_complexity_summary.csv')

    # Summary for ACGMT depth sweep
    depth_summary = df_depth.groupby(['dataset', 'model']).agg(
        test_r2_mean=('test_r2', 'mean'),
        test_r2_std=('test_r2', 'std'),
        test_mse_mean=('test_mse', 'mean'),
        fit_time_mean=('fit_time_s', 'mean'),
    ).reset_index()
    save_results(depth_summary, 'results/acgmt_depth_summary.csv')

    print("\n=== MAIN COMPARISON SUMMARY ===")
    print(summary.to_string(index=False))

    print("\n=== GBR COMPLEXITY SWEEP SUMMARY ===")
    print(gbr_summary.to_string(index=False))

    print("\n=== ACGMT DEPTH SWEEP SUMMARY ===")
    print(depth_summary.to_string(index=False))

    return df_main, df_gbr, df_depth


if __name__ == '__main__':
    run_experiments()
