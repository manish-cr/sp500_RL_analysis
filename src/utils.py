import os
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

# Data loading and preparation utilities

def load_sp500_data(csv_path):
    df = pd.read_csv(csv_path, parse_dates=["Date"])
    df = df.rename(columns={"S&P500": "Price"})
    df = df.set_index("Date")
    prices = df["Price"].values.reshape(-1, 1)
    scaler = MinMaxScaler()
    prices_scaled = scaler.fit_transform(prices)
    dates = df.index
    return prices_scaled, dates, scaler

def create_sequences(data, dates, input_len=60, output_len=30):
    X, y = [], []
    X_dates, y_dates = [], []
    for i in range(len(data) - input_len - output_len):
        X.append(data[i:i+input_len])
        y.append(data[i+input_len:i+input_len+output_len])
        X_dates.append(dates[i+input_len])
        y_dates.append(dates[i+input_len:i+input_len+output_len])
    return torch.tensor(X), torch.tensor(y), X_dates, y_dates

def train_test_split_sequences(X, y, split_ratio=0.75):
    split = int(split_ratio * len(X))
    X_train, y_train = X[:split], y[:split]
    X_test, y_test = X[split:], y[split:]
    return X_train, y_train, X_test, y_test
