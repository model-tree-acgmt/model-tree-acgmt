"""Main comparison experiments: ACGMT vs baselines across all datasets."""

import os
import numpy as np
from joblib import Parallel, delayed
from sklearn.linear_model import Ridge
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor

from code import RecursiveHybridModel
from acgmt import AdaptiveCoefficientGuidedModelTree
from datasets import ALL_DATASETS
from utils import evaluate_model, time_fit_predict, results_to_dataframe, save_results


def make_models(seed=42):
    """Model factories. seed is passed so ensemble randomness varies across runs."""
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


def _run_one(ds_name, ds_func, seed, model_name, model_factory):
    """Run a single (dataset, seed, model) combination."""
    (X_train, X_test, y_train, y_test), names = ds_func(random_state=seed)
    model = model_factory()
    try:
        timing = time_fit_predict(model, X_train, y_train, X_test)
    except Exception as e:
        print(f"  SKIP {model_name} on {ds_name} seed={seed}: {e}")
        return None

    # model is already fitted by time_fit_predict; reuse it for metrics
    metrics = evaluate_model(model, X_train, y_train, X_test, y_test)

    return {
        'dataset': ds_name,
        'model': model_name,
        'seed': seed,
        **metrics,
        **timing,
    }


def run_experiments(seeds=(42, 123, 456, 789, 1024), n_jobs=-1):
    os.makedirs('results', exist_ok=True)

    # Pre-fetch OpenML datasets once so parallel workers hit the cache
    for ds_func in ALL_DATASETS.values():
        ds_func(random_state=42)

    # Build list of jobs
    jobs = []
    for ds_name, ds_func in ALL_DATASETS.items():
        for seed in seeds:
            for model_name, model_factory in make_models(seed=seed).items():
                jobs.append(delayed(_run_one)(
                    ds_name, ds_func, seed, model_name, model_factory))

    print(f"Running {len(jobs)} jobs with n_jobs={n_jobs} ...")
    results = Parallel(n_jobs=n_jobs, verbose=1)(jobs)
    all_results = [r for r in results if r is not None]

    df = results_to_dataframe(all_results)
    save_results(df, 'results/comparison_full.csv')

    # Summary table: mean across seeds
    summary = df.groupby(['dataset', 'model']).agg(
        test_mse_mean=('test_mse', 'mean'),
        test_mse_std=('test_mse', 'std'),
        test_r2_mean=('test_r2', 'mean'),
        fit_time_mean=('fit_time_s', 'mean'),
    ).reset_index()
    save_results(summary, 'results/comparison_summary.csv')
    print("\n=== SUMMARY ===")
    print(summary.to_string(index=False))
    return df


if __name__ == '__main__':
    run_experiments()
