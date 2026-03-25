import time
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error


def evaluate_model(model, X_train, y_train, X_test, y_test):
    train_pred = model.predict(X_train)
    test_pred = model.predict(X_test)
    return {
        'train_mse': mean_squared_error(y_train, train_pred),
        'test_mse': mean_squared_error(y_test, test_pred),
        'train_r2': r2_score(y_train, train_pred),
        'test_r2': r2_score(y_test, test_pred),
        'train_mae': mean_absolute_error(y_train, train_pred),
        'test_mae': mean_absolute_error(y_test, test_pred),
    }


def time_fit_predict(model, X_train, y_train, X_test):
    t0 = time.perf_counter()
    model.fit(X_train, y_train)
    fit_time = time.perf_counter() - t0

    t0 = time.perf_counter()
    model.predict(X_test)
    predict_time = time.perf_counter() - t0

    return {
        'fit_time_s': fit_time,
        'predict_time_s': predict_time,
    }


def results_to_dataframe(results_list):
    return pd.DataFrame(results_list)


def save_results(df, path):
    df.to_csv(path, index=False)
    print(f"Results saved to {path}")
