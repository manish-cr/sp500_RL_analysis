## Requirements

All required packages are listed in `requirements.txt`. Main packages:

- numpy
- pandas
- matplotlib
- seaborn
- scikit-learn
- statsmodels
- torch, torchvision, torchaudio
- kagglehub
- cml (for GitHub Actions/CML reporting)
- jupyter (optional, for notebooks)

Install with:
```bash
pip install -r requirements.txt
```
# S&P 500 Stock Price Forecasting

## Folder Structure
```
.
└── src
    ├── script.ipynb           # Main notebook (data → models → forecast)
    ├── baseline_script.ipynb  # Simplified version
    ├── best_*.pth             # Trained model weights (RNN, LSTM, GRU)

# S&P 500 Time Series Project

## Project Structure

- `src/` - Source code for data loading, model training, and evaluation
  - `script.py` - Main script to run the workflow
  - `utils.py`, `baselines.py`, `models.py`, `print_utils.py` - Modularized code
  - `viz/` - All plots and visualization code are now in this folder
     - `plotting.py` - Visualization functions
     - All output plots are saved here as PNG files

## Usage

1. Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```
2. Run the main script:
    ```bash
    python src/script.py
    ```
    This will:
    - Download the S&P 500 dataset using kagglehub
    - Train baseline (ARIMA, ETS) and deep learning (RNN, GRU, LSTM) models
    - Print and save model comparison results
    - Save all plots to `src/viz/` as PNG files
    - Append the model comparison table to `report.md`

3. (Optional) Use CML for automated reporting in GitHub:
    - The workflow in `.github/workflows/cml.yaml` will run on every push.
    - It will post a comment with the model comparison table and plots from `src/viz/` to your PR or commit.

## Output

- All plots are saved in `src/viz/`:
  - `comparison_rmse.png`, `comparison_r2.png`, `sample_predictions.png`
- Model comparison table and summary are appended to `report.md`.

## Notes

- Make sure you have a valid Kaggle API token for kagglehub to work.
- The workflow assumes all scripts and folders are under `src/`.
- You can customize the CML workflow and report as needed.
```

## What It Does
1. **Downloads** S&P 500 daily prices (2014-present) from Kaggle
2. **Preprocesses** data and creates time-series features
3. **Trains 3 neural networks**: RNN, LSTM, GRU for price prediction
4. **Evaluates** models using RMSE, MAE, R² metrics
5. **Forecasts** future stock prices with uncertainty estimates

## Setup
```bash
pip install kagglehub pandas matplotlib scikit-learn torch
# Configure Kaggle API token at ~/.kaggle/kaggle.json
```

## Run Options
- **Interactive**: Open `script.ipynb` in Jupyter (recommended)
- **Quick test**: Run baseline notebook for simpler version
- **Use trained models**: Load `.pth` files for immediate predictions

## Key Results
- GRU typically performs best for this time-series task
- All models saved for quick inference without retraining
- Visualizations show predictions vs actual prices

## Models Available
- `best_rnn.pth` - Basic Recurrent Neural Network
- `best_lstm.pth` - Long Short-Term Memory (usually best)
- `best_gru.pth` - Gated Recurrent Unit

## Notes
- Data auto-downloads from Kaggle on first run
- Models can be retrained by adjusting hyperparameters in notebook
- GPU recommended but not required