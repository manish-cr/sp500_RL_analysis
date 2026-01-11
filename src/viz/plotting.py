import matplotlib.pyplot as plt
import numpy as np
import torch
import pandas as pd

def plot_comparison_and_samples(comparison_data, baseline_results, nn_results, y_test, sample_idx=0):
    import os
    # Ensure viz directory exists
    save_dir = os.path.dirname(os.path.abspath(__file__))
    # Bar plots for RMSE and R²
    models = [row['Model'] for row in comparison_data]
    rmses = [row['RMSE'] for row in comparison_data]
    r2s = [row['R²'] for row in comparison_data]
    colors = ['orange', 'purple', 'red', 'green', 'blue']
    # RMSE plot
    plt.figure(figsize=(12, 5))
    plt.bar(models, rmses, color=colors[:len(models)])
    plt.title('Model Comparison: RMSE (Lower is better)')
    plt.ylabel('RMSE')
    plt.xticks(rotation=45)
    plt.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'comparison_rmse.png'))
    plt.close()

    # R2 plot
    plt.figure(figsize=(12, 5))
    plt.bar(models, r2s, color=colors[:len(models)])
    plt.title('Model Comparison: R² Score (Higher is better)')
    plt.ylabel('R² Score')
    plt.xticks(rotation=45)
    plt.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'comparison_r2.png'))
    plt.close()

    # Sample predictions
    plt.figure(figsize=(15, 10))
    nn_models = list(nn_results.keys())
    for idx, model_name in enumerate(nn_models):
        plt.subplot(2, 3, idx+1)
        result = nn_results[model_name]
        actual = y_test[sample_idx, :, 0].cpu().numpy() if torch.is_tensor(y_test) else y_test[sample_idx, :, 0]
        predicted = result['predictions'][sample_idx, :, 0].cpu().numpy() if torch.is_tensor(result['predictions']) else result['predictions'][sample_idx, :, 0]
        nn_colors = {'RNN': 'red', 'GRU': 'green', 'LSTM': 'blue'}
        plt.plot(actual, label='Actual', marker='o', alpha=0.7, linewidth=2)
        plt.plot(predicted, label='Predicted', marker='x', color=nn_colors.get(model_name, 'black'), alpha=0.7, linewidth=2)
        plt.xlabel('Time Step')
        plt.ylabel('Normalized Price')
        plt.title(f'{model_name} Prediction')
        plt.legend()
        plt.grid(True, alpha=0.3)
    baseline_models = list(baseline_results.keys())
    for idx, model_name in enumerate(baseline_models):
        plt.subplot(2, 3, idx+4)
        result = baseline_results[model_name]
        actual = result['actuals'][sample_idx]
        predicted = result['predictions'][sample_idx]
        base_colors = {'ARIMA': 'orange', 'ETS': 'purple'}
        plt.plot(actual, label='Actual', marker='o', alpha=0.7, linewidth=2)
        plt.plot(predicted, label='Predicted', marker='x', color=base_colors.get(model_name, 'brown'), alpha=0.7, linewidth=2)
        plt.xlabel('Time Step')
        plt.ylabel('Normalized Price')
        plt.title(f'{model_name} Prediction')
        plt.legend()
        plt.grid(True, alpha=0.3)
    plt.suptitle('Sample Predictions Comparison (First Time Series)', fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'sample_predictions.png'))
    plt.close()
