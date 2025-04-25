import pandas as pd
import numpy as np
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.statespace.sarimax import SARIMAX
from pmdarima.arima import auto_arima
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

features = pd.read_csv("walmart-recruiting-store-sales-forecasting/features.csv", parse_dates=['Date'])
stores = pd.read_csv("walmart-recruiting-store-sales-forecasting/stores.csv")
train = pd.read_csv("walmart-recruiting-store-sales-forecasting/train.csv", parse_dates=['Date'])

features = pd.merge(features, stores, on='Store', how='left')
df = pd.merge(train, features, on=['Store', 'Date', 'IsHoliday'], how='left')
df = df.sort_values(by='Date')
markdown_cols = ['MarkDown1', 'MarkDown2', 'MarkDown3', 'MarkDown4', 'MarkDown5']
df[markdown_cols] = df[markdown_cols].fillna(0)

df_agg = df.groupby('Date')['Weekly_Sales'].sum().reset_index()
df_agg.set_index('Date', inplace=True)

adf_result = adfuller(df_agg['Weekly_Sales'])
adf_result_pvalue = adf_result[1]

test_weeks = 10
train_series = df_agg[:-test_weeks]['Weekly_Sales']
test_series = df_agg[-test_weeks:]['Weekly_Sales']
test_dates = df_agg[-test_weeks:].index

# ARIMA
arima_model = auto_arima(train_series, seasonal=False, stepwise=True, suppress_warnings=True, trace=True)
arima_result = arima_model.fit(train_series)
arima_forecast = arima_model.predict(n_periods=test_weeks)

# SARIMA
sarima_model = auto_arima(train_series, seasonal=True, m=52, stepwise=True, suppress_warnings=True, trace=True)
sarima_result = sarima_model.fit(train_series)
sarima_forecast = sarima_model.predict(n_periods=test_weeks)

def evaluate_model(true, pred, train_true, train_pred, model_name):
    val_mse = mean_squared_error(true, pred)
    val_rmse = np.sqrt(val_mse)
    val_r2 = r2_score(true, pred)

    train_mse = mean_squared_error(train_true, train_pred)
    train_rmse = np.sqrt(train_mse)
    train_r2 = r2_score(train_true, train_pred)

    print(f"\n {model_name} Performance:")
    print(f"Train RMSE: {train_rmse:.2f} | Train R²: {train_r2:.4f}")
    print(f"Val   RMSE: {val_rmse:.2f} | Val   R²: {val_r2:.4f}")

arima_train_pred = arima_model.predict_in_sample()
sarima_train_pred = sarima_model.predict_in_sample()

evaluate_model(test_series, arima_forecast, train_series, arima_train_pred, "ARIMA")
evaluate_model(test_series, sarima_forecast, train_series, sarima_train_pred, "SARIMA")

# Plotting
plt.figure(figsize=(14, 6))
plt.plot(df_agg.index, df_agg['Weekly_Sales'], label='Historical Sales')
plt.plot(test_dates, arima_forecast, label='ARIMA Forecast', linestyle='--', marker='o')
plt.plot(test_dates, sarima_forecast, label='SARIMA Forecast', linestyle='--', marker='x')
plt.axvline(test_dates[0], color='gray', linestyle=':', label='Train/Test Split')
plt.title("ARIMA vs SARIMA - Forecast vs Actual")
plt.xlabel("Date")
plt.ylabel("Weekly Sales")
plt.legend()
plt.grid()
plt.show()

adf_result_pvalue, arima_model.order, sarima_model.order, sarima_model.seasonal_order
