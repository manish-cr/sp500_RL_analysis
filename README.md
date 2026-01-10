# S&P 500 Stock Price Forecasting

## Folder Structure
```
.
└── src
    ├── script.ipynb           # Main notebook (data → models → forecast)
    ├── baseline_script.ipynb  # Simplified version
    ├── best_*.pth             # Trained model weights (RNN, LSTM, GRU)
    └── sp500_data.pkl         # Preprocessed S&P 500 data
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
- LSTM typically performs best for this time-series task
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