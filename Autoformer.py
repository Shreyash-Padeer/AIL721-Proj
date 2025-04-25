import pandas as pd
import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, r2_score

train_df = pd.read_csv('walmart-recruiting-store-sales-forecasting/train.csv')
features_df = pd.read_csv('walmart-recruiting-store-sales-forecasting/features.csv')
stores_df = pd.read_csv('walmart-recruiting-store-sales-forecasting/stores.csv')

walmart_store_group = train_df.groupby(["Store", "Date"])[["Weekly_Sales"]].sum().reset_index()
result = pd.merge(walmart_store_group, stores_df, on='Store')
data = pd.merge(result, features_df, on=['Store', 'Date'])
data['IsHoliday'] = data['IsHoliday'].apply(lambda x: 1 if x else 0)
data["Date"] = pd.to_datetime(data.Date)
data["Day"] = data.Date.dt.day
data["Month"] = data.Date.dt.month
data["Year"] = data.Date.dt.year
data['Week'] = data.Date.dt.isocalendar().week
data = data.drop(columns=["MarkDown1", "MarkDown2", "MarkDown3", "MarkDown4", "MarkDown5"])

scaler = MinMaxScaler()
scaled_cols = ["Weekly_Sales", "Temperature", "Fuel_Price", "CPI", "Unemployment", "Size"]
data[scaled_cols] = scaler.fit_transform(data[scaled_cols])

feature_cols = ["Store", "Temperature", "Fuel_Price", "CPI", "Unemployment", "IsHoliday", "Size"]
target_col = "Weekly_Sales"
SEQ_LEN = 8

class SalesDataset(Dataset):
    def __init__(self, df, seq_len):
        self.X, self.y = [], []
        for store in df["Store"].unique():
            store_df = df[df["Store"] == store].sort_values("Date")
            store_data = store_df[feature_cols + [target_col]].values
            for i in range(len(store_data) - seq_len):
                x_seq = store_data[i:i + seq_len, :-1]
                y_val = store_data[i + seq_len, -1]
                self.X.append(x_seq)
                self.y.append(y_val)
        self.X = torch.tensor(self.X, dtype=torch.float32)
        self.y = torch.tensor(self.y, dtype=torch.float32).unsqueeze(-1)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

dataset = SalesDataset(data, SEQ_LEN)
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_ds, val_ds = torch.utils.data.random_split(dataset, [train_size, val_size])
train_dl = DataLoader(train_ds, batch_size=64, shuffle=False)
val_dl = DataLoader(val_ds, batch_size=64, shuffle=False)

class RNNModel(nn.Module):
    def __init__(self, input_size, hidden_size=64):
        super().__init__()
        self.rnn = nn.RNN(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        out, _ = self.rnn(x)
        return self.fc(out[:, -1, :])


class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size=64):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])


class TransformerModel(nn.Module):
    def __init__(self, input_size, embed_dim=16, nhead=4, num_layers=2, dim_feedforward=128):
        super().__init__()
        self.input_proj = nn.Linear(input_size, embed_dim)
        encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=nhead, dim_feedforward=dim_feedforward, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc = nn.Linear(embed_dim, 1)

    def forward(self, x):
        x = self.input_proj(x)
        out = self.transformer(x)
        return self.fc(out[:, -1, :])


class AutoformerModel(nn.Module):
    def __init__(self, input_size, d_model=64, nhead=4, num_layers=2, dim_feedforward=128):
        super().__init__()
        self.input_proj = nn.Linear(input_size, d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.seasonal_fc = nn.Linear(d_model, 1)
        self.trend_fc = nn.Linear(d_model, 1)

    def forward(self, x):
        x_proj = self.input_proj(x)
        enc_out = self.encoder(x_proj)

        last_hidden = enc_out[:, -1, :]

        seasonal_out = self.seasonal_fc(last_hidden)
        trend_out = self.trend_fc(last_hidden)

        return seasonal_out + trend_out

class CNNLSTMModel(nn.Module):
    def __init__(self, input_size, cnn_channels=32, lstm_hidden_size=64):
        super().__init__()
        self.cnn = nn.Conv1d(in_channels=input_size, out_channels=cnn_channels, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.lstm = nn.LSTM(input_size=cnn_channels, hidden_size=lstm_hidden_size, batch_first=True)
        self.fc = nn.Linear(lstm_hidden_size, 1)

    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = self.relu(self.cnn(x))
        x = x.permute(0, 2, 1)
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])


def train_model(model, train_dl, val_dl, epochs=10, lr=1e-3):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    for epoch in range(epochs):
        model.train()
        train_losses = []
        train_preds, train_targets = [], []
        for xb, yb in train_dl:
            xb, yb = xb.to(device), yb.to(device)
            pred = model(xb)
            loss = criterion(pred, yb)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())

            train_preds.append(pred.detach().cpu())
            train_targets.append(yb.detach().cpu())

        train_preds = torch.cat(train_preds).squeeze()
        train_targets = torch.cat(train_targets).squeeze()
        train_mse = mean_squared_error(train_targets, train_preds)
        train_r2 = r2_score(train_targets, train_preds)

        model.eval()
        val_preds, val_targets = [], []
        with torch.no_grad():
            for xb, yb in val_dl:
                xb, yb = xb.to(device), yb.to(device)
                pred = model(xb)
                val_preds.append(pred.cpu())
                val_targets.append(yb.cpu())

        val_preds = torch.cat(val_preds).squeeze()
        val_targets = torch.cat(val_targets).squeeze()
        val_mse = mean_squared_error(val_targets, val_preds)
        val_r2 = r2_score(val_targets, val_preds)

        print(f"Epoch {epoch+1} | Train Loss: {np.mean(train_losses):.4f} | Train MSE: {train_mse:.4f} | Train R²: {train_r2:.4f} | Val MSE: {val_mse:.4f} | Val R²: {val_r2:.4f}")

model = AutoformerModel(input_size=len(feature_cols))

print(f"\nTraining Autoformer model")
train_model(model, train_dl, val_dl, epochs=60)
