
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Bidirectional
import random
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Dropout, Conv1D, LayerNormalization, Activation
from tensorflow.keras.layers import add, GlobalAveragePooling1D, SpatialDropout1D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, Callback


random.seed(42)
np.random.seed(42)
tf.random.set_seed(42)


train_df=pd.read_csv('/home/scai/mtech/aib242288/scratch/AIL721-DL/Project/Walmart Data/train.csv')
features_df=pd.read_csv('/home/scai/mtech/aib242288/scratch/AIL721-DL/Project/Walmart Data/features.csv')
stores_df=pd.read_csv('/home/scai/mtech/aib242288/scratch/AIL721-DL/Project/Walmart Data/stores.csv')


walmart_store_group=train_df.groupby(["Store","Date"])[["Weekly_Sales"]].sum()
walmart_store_group.reset_index(inplace=True)


result = pd.merge(walmart_store_group, stores_df, how='inner', on='Store', left_on=None, right_on=None,
        left_index=False, right_index=False, sort=False,
        suffixes=('_x', '_y'), copy=True, indicator=False)

data = pd.merge(result, features_df, how='inner', on=['Store','Date'], left_on=None, right_on=None,
        left_index=False, right_index=False, sort=False,
        suffixes=('_x', '_y'), copy=True, indicator=False)


data['IsHoliday'] = data['IsHoliday'].apply(lambda x: 1 if x == True else 0)


data["Date"]=pd.to_datetime(data.Date)


data["Day"]=data.Date.dt.day
data["Month"]=data.Date.dt.month
data["Year"]=data.Date.dt.year


data['Week'] = data.Date.dt.isocalendar().week

data = data.drop(columns=["MarkDown1", "MarkDown2",	"MarkDown3", "MarkDown4", "MarkDown5"	])

train_data = data


features = ['Store', 'IsHoliday', 'Temperature', 'Fuel_Price', 'CPI', 'Unemployment', 'Type', 'Size', 'Year', 'Month', 'Day', 'Week']
X = train_data[features]
X = pd.get_dummies(X, columns=['Type'], drop_first=True)
X.fillna(0, inplace=True)
y = train_data['Weekly_Sales']

scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)
y_scaled = scaler.fit_transform(y.values.reshape(-1, 1))

sequence_length = 15
X_bilstm = []
y_bilstm = []
for i in range(sequence_length, len(X_scaled)):
    X_bilstm.append(X_scaled[i-sequence_length:i])
    y_bilstm.append(y_scaled[i])

X_bilstm, y_bilstm = np.array(X_bilstm), np.array(y_bilstm)

X_train, X_test, y_train, y_test = train_test_split(
    X_bilstm, y_bilstm, test_size=0.2, random_state=42, shuffle=False)

model = Sequential([
    Bidirectional(LSTM(100, activation='tanh', return_sequences=True)),
    Dropout(0.2),
    Bidirectional(LSTM(50, activation='tanh')),
    Dropout(0.2),
    Dense(1)
])

model.compile(optimizer='adam', loss='mse')

early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

history = model.fit(
    X_train, y_train,
    epochs=15,
    batch_size=32,
    validation_split=0.1,
    callbacks=[early_stop],
    verbose=1
)

y_train_pred = model.predict(X_train)
y_test_pred = model.predict(X_test)

y_train_pred = scaler.inverse_transform(y_train_pred)
y_test_pred = scaler.inverse_transform(y_test_pred)
y_train = scaler.inverse_transform(y_train)
y_test = scaler.inverse_transform(y_test)

train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
train_r2 = r2_score(y_train, y_train_pred)
test_r2 = r2_score(y_test, y_test_pred)

print("\nBidirectional LSTM (BiLSTM) Model Performance:")
print(f"Train RMSE: {train_rmse:.2f}")
print(f"Test RMSE: {test_rmse:.2f}")
print(f"Train R²: {train_r2:.4f}")
print(f"Test R²: {test_r2:.4f}")