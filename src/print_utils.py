import numpy as np
import torch

def print_comparison_header():
    print("Starting complete model comparison...")
    print("="*60)

def print_data_shapes(X_train, y_train, X_test, y_test):
    print(f"Data shapes:")
    print(f"X_train: {tuple(X_train.shape)}, y_train: {tuple(y_train.shape)}")
    print(f"X_test: {tuple(X_test.shape)}, y_test: {tuple(y_test.shape)}\n")

def print_phase_header(phase):
    print("="*60)
    print(phase)
    print("="*60)

def print_baseline_results(results):
    for name, result in results.items():
        metrics = result['metrics']
        print(f"\n{name} Results:")
        print(f"  RMSE: {metrics['RMSE']:.6f}")
        print(f"  MAE:  {metrics['MAE']:.6f}")
        print(f"  R²:   {metrics['R²']:.6f}")
        print(f"  Successfully evaluated {metrics['num_series']}/10 series")

def print_nn_results(nn_results):
    for name, result in nn_results.items():
        metrics = result['metrics']
        print(f"{name} Results:")
        print(f"  RMSE: {metrics['RMSE']:.6f}")
        print(f"  R²:   {metrics['R²']:.6f}")

def print_final_comparison(comparison_data):
    lines = []
    lines.append("\n" + "="*60)
    lines.append("FINAL COMPARISON: ALL MODELS")
    lines.append("="*60)
    lines.append("\nModel Performance Comparison:")
    lines.append("-" * 80)
    lines.append(f"{'Model':<15} {'RMSE':<15} {'R²':<15} {'Series Evaluated':<20}")
    lines.append("-" * 80)
    for row in comparison_data:
        lines.append(f"{row['Model']:<15} {row['RMSE']:<15.6f} {row['R²']:<15.6f} {row['Series']:<20}")
    lines.append("-" * 80)
    best = min(comparison_data, key=lambda x: x['RMSE'])
    lines.append(f"\nBest model by RMSE: {best['Model']}")
    lines.append(f"RMSE: {best['RMSE']:.6f}")
    lines.append(f"R²: {best['R²']:.6f}")
    # Print to console
    for l in lines:
        print(l)
    # Also write to report.md (append)
    with open("report.md", "a") as f:
        for l in lines:
            f.write(l + "\n")


