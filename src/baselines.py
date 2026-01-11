import numpy as np
import torch
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.holtwinters import ExponentialSmoothing
import warnings

warnings.filterwarnings('ignore')

def run_statsmodels_baseline(X_test, y_test, n_series=10):
    """
    Run ARIMA and Exponential Smoothing baselines on the first n_series of the test set.
    Returns a dict with predictions and metrics for each model.
    """
    X_test_np = X_test.numpy().squeeze(-1)
    y_test_np = y_test.numpy().squeeze(-1)
    results = {}
    n_series = min(n_series, len(X_test_np))
    actual_array = y_test_np[:n_series, :30]

    # ARIMA
    arima_predictions = []
    for i in range(n_series):
        try:
            model = ARIMA(X_test_np[i], order=(1, 1, 1))
            model_fit = model.fit()
            forecast = model_fit.forecast(steps=30)
            arima_predictions.append(forecast)
        except Exception:
            arima_predictions.append(np.full(30, np.nan))
    arima_array = np.array(arima_predictions)
    valid_mask = ~np.isnan(arima_array).any(axis=1)
    if valid_mask.sum() > 0:
        valid_arima = arima_array[valid_mask]
        valid_actual = actual_array[valid_mask]
        rmse = np.sqrt(np.mean((valid_arima - valid_actual) ** 2))
        mae = np.mean(np.abs(valid_arima - valid_actual))
        ss_res = np.sum((valid_actual - valid_arima) ** 2)
        ss_tot = np.sum((valid_actual - np.mean(valid_actual)) ** 2)
        r2 = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0
        results['ARIMA'] = {
            'predictions': arima_array,
            'actuals': actual_array,
            'metrics': {'RMSE': rmse, 'MAE': mae, 'R²': r2, 'num_series': valid_mask.sum()}
        }
    # ETS
    ets_predictions = []
    for i in range(n_series):
        try:
            model = ExponentialSmoothing(X_test_np[i], trend='add', seasonal=None)
            model_fit = model.fit()
            forecast = model_fit.forecast(steps=30)
            ets_predictions.append(forecast)
        except Exception:
            ets_predictions.append(np.full(30, np.nan))
    ets_array = np.array(ets_predictions)
    valid_mask = ~np.isnan(ets_array).any(axis=1)
    if valid_mask.sum() > 0:
        valid_ets = ets_array[valid_mask]
        valid_actual = actual_array[valid_mask]
        rmse = np.sqrt(np.mean((valid_ets - valid_actual) ** 2))
        mae = np.mean(np.abs(valid_ets - valid_actual))
        ss_res = np.sum((valid_actual - valid_ets) ** 2)
        ss_tot = np.sum((valid_actual - np.mean(valid_actual)) ** 2)
        r2 = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0
        results['ETS'] = {
            'predictions': ets_array,
            'actuals': actual_array,
            'metrics': {'RMSE': rmse, 'MAE': mae, 'R²': r2, 'num_series': valid_mask.sum()}
        }
    return results
