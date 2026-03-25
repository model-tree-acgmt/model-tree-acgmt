"""Ablation experiments: isolate effect of each ACGMT enhancement."""

import os
import numpy as np
from acgmt import AdaptiveCoefficientGuidedModelTree
from datasets import ALL_DATASETS
from utils import evaluate_model, time_fit_predict, results_to_dataframe, save_results


def run_ablation(name, configs, seeds=(42, 123, 456, 789, 1024)):
    """Run one ablation axis across all datasets."""
    all_results = []
    for ds_name, ds_func in ALL_DATASETS.items():
        for seed in seeds:
            (X_train, X_test, y_train, y_test), _ = ds_func(random_state=seed)
            for cfg_name, kwargs in configs.items():
                model = AdaptiveCoefficientGuidedModelTree(
                    max_depth=4, min_samples_split=100, random_state=seed, **kwargs)
                timing = time_fit_predict(model, X_train, y_train, X_test)
                model2 = AdaptiveCoefficientGuidedModelTree(
                    max_depth=4, min_samples_split=100, random_state=seed, **kwargs)
                model2.fit(X_train, y_train)
                metrics = evaluate_model(model2, X_train, y_train, X_test, y_test)
                all_results.append({
                    'dataset': ds_name, 'config': cfg_name, 'seed': seed,
                    **metrics, **timing,
                })
    df = results_to_dataframe(all_results)
    summary = df.groupby(['dataset', 'config']).agg(
        test_mse_mean=('test_mse', 'mean'),
        test_mse_std=('test_mse', 'std'),
        test_r2_mean=('test_r2', 'mean'),
        fit_time_mean=('fit_time_s', 'mean'),
    ).reset_index()
    save_results(df, f'results/ablation_{name}_full.csv')
    save_results(summary, f'results/ablation_{name}_summary.csv')
    print(f"\n=== Ablation: {name} ===")
    print(summary.to_string(index=False))
    return summary


def main():
    os.makedirs('results', exist_ok=True)

    # A. Top-k ablation (standardize=True, no adaptive grid, no pruning, full data)
    print("Running top-k ablation...")
    run_ablation('topk', {
        f'top-{k}': dict(top_k=k, standardize=True, adaptive_grid=False,
                         prune=False, val_fraction=0)
        for k in [1, 2, 3, 5]
    })

    # B. Standardization ablation (top_k=3, no adaptive grid, no pruning, full data)
    print("\nRunning standardization ablation...")
    run_ablation('standardize', {
        'no-std': dict(top_k=3, standardize=False, adaptive_grid=False,
                       prune=False, val_fraction=0),
        'with-std': dict(top_k=3, standardize=True, adaptive_grid=False,
                         prune=False, val_fraction=0),
    })

    # C. Adaptive grid ablation (top_k=3, std=True, no pruning, full data)
    print("\nRunning adaptive grid ablation...")
    run_ablation('grid', {
        'fixed-grid': dict(top_k=3, standardize=True, adaptive_grid=False,
                           prune=False, val_fraction=0),
        'adaptive-grid': dict(top_k=3, standardize=True, adaptive_grid=True,
                              prune=False, val_fraction=0),
    })

    # D. Pruning ablation (top_k=3, std=True, adaptive=True)
    # Both conditions use val_fraction=0.2 so the tree is built on the same
    # 80% subset.  Only the pruning step differs.
    print("\nRunning pruning ablation...")
    run_ablation('pruning', {
        'no-prune': dict(top_k=3, standardize=True, adaptive_grid=True,
                         prune=False, val_fraction=0.2),
        'with-prune': dict(top_k=3, standardize=True, adaptive_grid=True,
                           prune=True, val_fraction=0.2),
    })


if __name__ == '__main__':
    main()
