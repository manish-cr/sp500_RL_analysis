import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import numpy as np

class SimpleRNNModel(nn.Module):
    def __init__(self, input_len=60, output_len=30):
        super().__init__()
        self.rnn = nn.RNN(
            input_size=1, hidden_size=50, num_layers=2, batch_first=True, dropout=0.2, nonlinearity='tanh')
        self.fc = nn.Sequential(
            nn.Linear(50, 100), nn.ReLU(), nn.Dropout(0.2), nn.Linear(100, output_len))
    def forward(self, x):
        rnn_out, hidden = self.rnn(x)
        last_hidden = hidden[-1]
        predictions = self.fc(last_hidden)
        return predictions.unsqueeze(-1)

class SimpleGRUModel(nn.Module):
    def __init__(self, input_len=60, output_len=30):
        super().__init__()
        self.gru = nn.GRU(
            input_size=1, hidden_size=50, num_layers=2, batch_first=True, dropout=0.2)
        self.fc = nn.Sequential(
            nn.Linear(50, 100), nn.ReLU(), nn.Dropout(0.2), nn.Linear(100, output_len))
    def forward(self, x):
        gru_out, hidden = self.gru(x)
        last_hidden = hidden[-1]
        predictions = self.fc(last_hidden)
        return predictions.unsqueeze(-1)

class SimpleLSTMModel(nn.Module):
    def __init__(self, input_len=60, output_len=30):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=1, hidden_size=50, num_layers=2, batch_first=True, dropout=0.2)
        self.fc = nn.Sequential(
            nn.Linear(50, 100), nn.ReLU(), nn.Dropout(0.2), nn.Linear(100, output_len))
    def forward(self, x):
        lstm_out, (hidden, cell) = self.lstm(x)
        last_hidden = hidden[-1]
        predictions = self.fc(last_hidden)
        return predictions.unsqueeze(-1)

def train_model(model, model_name, X_train, y_train, X_test, y_test, n_epochs=200, lr=0.001, batch_size=32, patience=30):
    print(f"\n{'='*60}")
    print(f"Training {model_name}")
    print(f"{'='*60}")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    model = model.to(device)
    X_train = X_train.float().to(device)
    y_train = y_train.float().to(device)
    X_test = X_test.float().to(device)
    y_test = y_test.float().to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=patience//2, factor=0.5)
    loss_fn = nn.MSELoss()
    train_dataset = data.TensorDataset(X_train, y_train)
    loader = data.DataLoader(train_dataset, shuffle=True, batch_size=batch_size, drop_last=False)
    best_test_loss = float('inf')
    patience_counter = 0
    train_losses = []
    test_losses = []
    for epoch in range(n_epochs):
        model.train()
        train_loss = 0
        for X_batch, y_batch in loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            y_pred = model(X_batch)
            loss = loss_fn(y_pred, y_batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_loss += loss.item()
        avg_train_loss = train_loss / len(loader)
        train_losses.append(avg_train_loss)
        model.eval()
        with torch.no_grad():
            y_pred_test = model(X_test)
            test_loss = loss_fn(y_pred_test, y_test)
            test_rmse = torch.sqrt(test_loss)
            test_losses.append(test_loss.item())
        scheduler.step(test_loss)
        if test_loss < best_test_loss:
            best_test_loss = test_loss
            patience_counter = 0
            torch.save({'epoch': epoch,'model_state_dict': model.state_dict(),'optimizer_state_dict': optimizer.state_dict(),'loss': best_test_loss}, f'best_{model_name.lower()}.pth')
        else:
            patience_counter += 1
        if epoch % 100 == 0:
            with torch.no_grad():
                y_pred_train = model(X_train)
                train_rmse = torch.sqrt(loss_fn(y_pred_train, y_train))
            print(f"Epoch {epoch:4d}: Train RMSE: {train_rmse:.4f}, Test RMSE: {test_rmse:.4f}, LR: {optimizer.param_groups[0]['lr']:.6f}")
        if patience_counter >= patience:
            print(f"Early stopping at epoch {epoch}")
            break
    checkpoint = torch.load(f'best_{model_name.lower()}.pth', map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    with torch.no_grad():
        y_pred_train = model(X_train)
        y_pred_test = model(X_test)
        train_rmse = torch.sqrt(loss_fn(y_pred_train, y_train)).item()
        test_rmse = torch.sqrt(loss_fn(y_pred_test, y_test)).item()
    print(f"\n{model_name} Final Results:")
    print(f"Train RMSE: {train_rmse:.6f}")
    print(f"Test RMSE:  {test_rmse:.6f}")
    return model, train_losses, test_losses, y_pred_test
