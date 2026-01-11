import os
import torch
import numpy as np
from utils import load_sp500_data, create_sequences, train_test_split_sequences
from baselines import run_statsmodels_baseline
from models import SimpleRNNModel, SimpleGRUModel, SimpleLSTMModel, train_model
from print_utils import (
    print_comparison_header, print_data_shapes, print_phase_header,
    print_baseline_results, print_nn_results, print_final_comparison
)

def main():
    print_comparison_header()
    # Data loading (use kagglehub to download dataset as in script.ipynb)
    import kagglehub
    path = kagglehub.dataset_download("andrewmvd/sp-500-stocks")
    print("Path to dataset files:", path)
    csv_path = os.path.join(path, "sp500_index.csv")
    prices_scaled, dates, scaler = load_sp500_data(csv_path)
    X, y, X_dates, y_dates = create_sequences(prices_scaled, dates, input_len=60, output_len=30)
    X_train, y_train, X_test, y_test = train_test_split_sequences(X, y, split_ratio=0.75)
    print_data_shapes(X_train, y_train, X_test, y_test)

    # Baseline models
    print_phase_header('PHASE 1: BASELINE MODELS (Statsmodels)')
    baseline_results = run_statsmodels_baseline(X_test, y_test, n_series=10)
    print_baseline_results(baseline_results)

    # Neural network models
    print_phase_header('PHASE 2: NEURAL NETWORK MODELS')
    nn_results = {}
    models_to_train = {
        'RNN': SimpleRNNModel(input_len=60, output_len=30),
        'GRU': SimpleGRUModel(input_len=60, output_len=30),
        'LSTM': SimpleLSTMModel(input_len=60, output_len=30)
    }
    for model_name, model in models_to_train.items():
        print(f"\nTraining {model_name}...")
        trained_model, train_losses, test_losses, y_pred_test = train_model(
            model, model_name, X_train, y_train, X_test, y_test,
            n_epochs=200, lr=0.001, batch_size=32, patience=30
        )
        with torch.no_grad():
            test_rmse = torch.sqrt(torch.nn.MSELoss()(y_pred_test, y_test)).item()
            y_test_np = y_test.cpu().numpy()
            y_pred_np = y_pred_test.cpu().numpy()
            ss_res = np.sum((y_test_np - y_pred_np) ** 2)
            ss_tot = np.sum((y_test_np - np.mean(y_test_np)) ** 2)
            r2 = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0
        nn_results[model_name] = {
            'model': trained_model,
            'train_losses': train_losses,
            'test_losses': test_losses,
            'predictions': y_pred_test,
            'metrics': {'RMSE': test_rmse, 'R²': r2, 'num_series': len(y_test)}
        }
        print_nn_results({model_name: nn_results[model_name]})

    # Final comparison
    comparison_data = []
    for name, result in baseline_results.items():
        metrics = result['metrics']
        comparison_data.append({
            'Model': name,
            'RMSE': metrics['RMSE'],
            'R²': metrics['R²'],
            'Series': metrics['num_series']
        })
    for name, result in nn_results.items():
        metrics = result['metrics']
        comparison_data.append({
            'Model': name,
            'RMSE': metrics['RMSE'],
            'R²': metrics['R²'],
            'Series': metrics['num_series']
        })
    print_final_comparison(comparison_data)
    # Visualization
    from viz.plotting import plot_comparison_and_samples
    plot_comparison_and_samples(comparison_data, baseline_results, nn_results, y_test, sample_idx=0)

if __name__ == "__main__":
    main()
