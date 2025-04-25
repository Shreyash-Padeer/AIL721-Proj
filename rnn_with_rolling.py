import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import SimpleRNN, Dense, Dropout
import random
import tensorflow as tf

# Set seeds for reproducibility
random.seed(42)
np.random.seed(42)
tf.random.set_seed(42)

# Load data
train_df = pd.read_csv('/home/scai/mtech/aib242288/scratch/AIL721-DL/Project/Walmart Data/train.csv')
features_df = pd.read_csv('/home/scai/mtech/aib242288/scratch/AIL721-DL/Project/Walmart Data/features.csv')
stores_df = pd.read_csv('/home/scai/mtech/aib242288/scratch/AIL721-DL/Project/Walmart Data/stores.csv')

# Merge and preprocess
walmart_store_group = train_df.groupby(["Store", "Date"])[["Weekly_Sales"]].sum().reset_index()
result = pd.merge(walmart_store_group, stores_df, how='inner', on='Store')
data = pd.merge(result, features_df, how='inner', on=['Store', 'Date'])

data['IsHoliday'] = data['IsHoliday'].astype(int)
data["Date"] = pd.to_datetime(data["Date"])
data["Day"] = data.Date.dt.day
data["Month"] = data.Date.dt.month
data["Year"] = data.Date.dt.year
data['Week'] = data.Date.dt.isocalendar().week

# Drop markdowns
data = data.drop(columns=["MarkDown1", "MarkDown2", "MarkDown3", "MarkDown4", "MarkDown5"])

# Sort and split BEFORE rolling
data = data.sort_values(['Store', 'Date']).reset_index(drop=True)
train_data, test_data = train_test_split(data, test_size=0.2, shuffle=False)

# Function to apply rolling features per store
def add_rolling_features(df):
    df = df.sort_values(['Store', 'Year', 'Month', 'Day']).reset_index(drop=True)
    stores = df['Store'].unique()
    result_df = pd.DataFrame()
    for store in stores:
        store_data = df[df['Store'] == store].copy()
        if 'Weekly_Sales' in store_data.columns:
            store_data['Sales_Roll_Mean_4W'] = store_data['Weekly_Sales'].rolling(window=4, min_periods=1).mean()
            store_data['Sales_Roll_Mean_8W'] = store_data['Weekly_Sales'].rolling(window=8, min_periods=1).mean()
            store_data['Sales_Roll_Mean_12W'] = store_data['Weekly_Sales'].rolling(window=12, min_periods=1).mean()
            store_data['Sales_Roll_Std_4W'] = store_data['Weekly_Sales'].rolling(window=4, min_periods=1).std().fillna(0)
            store_data['Sales_Roll_Std_8W'] = store_data['Weekly_Sales'].rolling(window=8, min_periods=1).std().fillna(0)
            store_data['Sales_Roll_Min_4W'] = store_data['Weekly_Sales'].rolling(window=4, min_periods=1).min()
            store_data['Sales_Roll_Max_4W'] = store_data['Weekly_Sales'].rolling(window=4, min_periods=1).max()
            store_data['Sales_Pct_Change_1W'] = store_data['Weekly_Sales'].pct_change(1).fillna(0)
            store_data['Sales_Expanding_Mean'] = store_data['Weekly_Sales'].expanding().mean()
        if 'Temperature' in store_data.columns:
            store_data['Temp_Roll_Mean_4W'] = store_data['Temperature'].rolling(window=4, min_periods=1).mean()
        if 'Fuel_Price' in store_data.columns:
            store_data['Fuel_Roll_Mean_4W'] = store_data['Fuel_Price'].rolling(window=4, min_periods=1).mean()
        store_data['Is_Q4'] = ((store_data['Month'] >= 10) & (store_data['Month'] <= 12)).astype(int)
        result_df = pd.concat([result_df, store_data], ignore_index=True)
    return result_df

# Apply rolling features to train and test separately to avoid leakage
train_data = add_rolling_features(train_data)
test_data = add_rolling_features(test_data)

# Fill NA
train_data.fillna(0, inplace=True)
test_data.fillna(0, inplace=True)

# Feature columns
features = ['Store', 'IsHoliday', 'Temperature', 'Fuel_Price', 'CPI', 'Unemployment',
            'Type', 'Size', 'Year', 'Month', 'Day', 'Week',
            'Sales_Roll_Mean_4W', 'Sales_Roll_Mean_8W', 'Sales_Roll_Mean_12W',
            'Sales_Roll_Std_4W', 'Sales_Roll_Std_8W', 'Sales_Roll_Min_4W', 'Sales_Roll_Max_4W',
            'Sales_Pct_Change_1W', 'Sales_Expanding_Mean', 'Temp_Roll_Mean_4W',
            'Fuel_Roll_Mean_4W', 'Is_Q4']

# Prepare inputs
X_train = pd.get_dummies(train_data[features], columns=['Type'], drop_first=True)
X_test = pd.get_dummies(test_data[features], columns=['Type'], drop_first=True)

# Align columns in case of missing dummies in test
X_train, X_test = X_train.align(X_test, join='left', axis=1, fill_value=0)

y_train = train_data['Weekly_Sales'].values.reshape(-1, 1)
y_test = test_data['Weekly_Sales'].values.reshape(-1, 1)

# Scale
scaler_X = MinMaxScaler()
X_train_scaled = scaler_X.fit_transform(X_train)
X_test_scaled = scaler_X.transform(X_test)

scaler_y = MinMaxScaler()
y_train_scaled = scaler_y.fit_transform(y_train)
y_test_scaled = scaler_y.transform(y_test)

# Create RNN sequences
sequence_length = 7
def create_sequences(X, y, seq_len):
    X_seq, y_seq = [], []
    for i in range(seq_len, len(X)):
        X_seq.append(X[i-seq_len:i])
        y_seq.append(y[i])
    return np.array(X_seq), np.array(y_seq)

X_train_seq, y_train_seq = create_sequences(X_train_scaled, y_train_scaled, sequence_length)
X_test_seq, y_test_seq = create_sequences(X_test_scaled, y_test_scaled, sequence_length)

# RNN Model
model = Sequential([
    SimpleRNN(100, activation='tanh', return_sequences=True, input_shape=(X_train_seq.shape[1], X_train_seq.shape[2])),
    Dropout(0.2),
    SimpleRNN(50, activation='tanh'),
    Dropout(0.2),
    Dense(1)
])

model.compile(optimizer='adam', loss='mse')

# Train
model.fit(X_train_seq, y_train_seq, epochs=10, batch_size=32)

# Predict
y_train_pred = model.predict(X_train_seq)
y_test_pred = model.predict(X_test_seq)

# Inverse scale
y_train_pred = scaler_y.inverse_transform(y_train_pred)
y_test_pred = scaler_y.inverse_transform(y_test_pred)
y_train_true = scaler_y.inverse_transform(y_train_seq)
y_test_true = scaler_y.inverse_transform(y_test_seq)

# Metrics
train_rmse = np.sqrt(mean_squared_error(y_train_true, y_train_pred))
test_rmse = np.sqrt(mean_squared_error(y_test_true, y_test_pred))
train_r2 = r2_score(y_train_true, y_train_pred)
test_r2 = r2_score(y_test_true, y_test_pred)

print(f"RNN Train RMSE: {train_rmse:.2f}")
print(f"RNN Test RMSE: {test_rmse:.2f}")
print(f"RNN Train R^2: {train_r2:.4f}")
print(f"RNN Test R^2: {test_r2:.4f}")
