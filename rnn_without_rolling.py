
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
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

sequence_length = 7
X_rnn = []
y_rnn = []
for i in range(sequence_length, len(X_scaled)):
  X_rnn.append(X_scaled[i - sequence_length:i])
  y_rnn.append(y_scaled[i])

X_rnn, y_rnn = np.array(X_rnn), np.array(y_rnn)

X_train_rnn, X_test_rnn, y_train_rnn, y_test_rnn = train_test_split(X_rnn, y_rnn, test_size=0.2, random_state=42, shuffle = False)

model = Sequential()
model.add(LSTM(50, activation='relu', input_shape=(X_train_rnn.shape[1], X_train_rnn.shape[2])))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mse')

model.fit(X_train_rnn, y_train_rnn, epochs=5, batch_size=32) 

y_train_pred_rnn = model.predict(X_train_rnn)
y_test_pred_rnn = model.predict(X_test_rnn)

y_train_pred_rnn = scaler.inverse_transform(y_train_pred_rnn)
y_test_pred_rnn = scaler.inverse_transform(y_test_pred_rnn)
y_train_rnn = scaler.inverse_transform(y_train_rnn)
y_test_rnn = scaler.inverse_transform(y_test_rnn)

train_rmse_rnn = np.sqrt(mean_squared_error(y_train_rnn, y_train_pred_rnn))
test_rmse_rnn = np.sqrt(mean_squared_error(y_test_rnn, y_test_pred_rnn))
train_r2_rnn = r2_score(y_train_rnn, y_train_pred_rnn)
test_r2_rnn = r2_score(y_test_rnn, y_test_pred_rnn)

print(f"RNN Train RMSE: {train_rmse_rnn}")
print(f"RNN Test RMSE: {test_rmse_rnn}")
print(f"RNN Train R^2: {train_r2_rnn}")
print(f"RNN Test R^2: {test_r2_rnn}")
