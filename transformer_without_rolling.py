import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Dropout, LayerNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
import tensorflow as tf
from tensorflow.keras import layers
import random


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



class TransformerBlock(layers.Layer):
    def __init__(self, embed_dim, num_heads, ff_dim, rate=0.1):
        super(TransformerBlock, self).__init__()
        self.att = layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
        self.ffn = tf.keras.Sequential([
            layers.Dense(ff_dim, activation="gelu"),  
            layers.Dense(embed_dim),
        ])
        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = layers.Dropout(rate)
        self.dropout2 = layers.Dropout(rate)

    def call(self, inputs, training=None):
        attn_output = self.att(inputs, inputs, training=training)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        return self.layernorm2(out1 + ffn_output)

class PositionalEmbedding(layers.Layer):
    def __init__(self, sequence_length, output_dim):
        super(PositionalEmbedding, self).__init__()

        
        position = np.arange(sequence_length)[:, np.newaxis]
        div_term = np.exp(np.arange(0, output_dim, 2) * -(np.log(10000.0) / output_dim))

        pe = np.zeros((sequence_length, output_dim))
        pe[:, 0::2] = np.sin(position * div_term)
        pe[:, 1::2] = np.cos(position * div_term)

        self.pe = tf.constant(pe.astype(np.float32))
        self.projection = layers.Dense(output_dim)

    def call(self, inputs):
        
        projected = self.projection(inputs)
        return projected + self.pe



sequence_length = 52
X_seq = []
y_seq = []
for i in range(sequence_length, len(X_scaled)):
    X_seq.append(X_scaled[i-sequence_length:i])
    y_seq.append(y_scaled[i])
X_seq, y_seq = np.array(X_seq), np.array(y_seq)


X_train, X_test, y_train, y_test = train_test_split(
    X_seq, y_seq, test_size=0.2, random_state=42, shuffle=False)


embed_dim = 64  
num_heads = 6   
ff_dim = 128    


inputs = Input(shape=(sequence_length, X_train.shape[2]))

x = layers.Dense(embed_dim)(inputs)
x = PositionalEmbedding(sequence_length, embed_dim)(x)
x = TransformerBlock(embed_dim, num_heads, ff_dim)(x)
x = layers.GlobalAveragePooling1D()(x)
x = Dropout(0.1)(x)
x = Dense(32, activation="relu")(x)
outputs = Dense(1)(x)

model = Model(inputs=inputs, outputs=outputs)
model.compile(optimizer=Adam(learning_rate=1e-3), loss='mse')


early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)


history = model.fit(
    X_train, y_train,
    batch_size=32,
    epochs=8,
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

print("\nTransformer Model Performance:")
print(f"Train RMSE: {train_rmse:.2f}")
print(f"Test RMSE: {test_rmse:.2f}")
print(f"Train R²: {train_r2:.4f}")
print(f"Test R²: {test_r2:.4f}")