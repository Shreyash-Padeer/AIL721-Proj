import pandas as pd
import numpy as np
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.arima.model import ARIMA
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

df_agg['RollingMean_4'] = df_agg['Weekly_Sales'].rolling(window=4).mean()
df_agg['RollingStd_4'] = df_agg['Weekly_Sales'].rolling(window=4).std()
df_agg.dropna(inplace=True)

test_weeks = 10
train_df = df_agg[:-test_weeks]
test_df = df_agg[-test_weeks:]
train_series = train_df['Weekly_Sales']
test_series = test_df['Weekly_Sales']
test_dates = test_df.index


model_arima = ARIMA(train_series, order=(5, 1, 2))
arima_result = model_arima.fit()
arima_forecast_result = arima_result.get_forecast(steps=test_weeks)
arima_forecast = arima_forecast_result.predicted_mean
arima_ci = arima_forecast_result.conf_int()

sarima_model = SARIMAX(
    train_series,
    order=(2, 1, 2),
    seasonal_order=(1, 1, 1, 52),
    exog=train_df[['RollingMean_4', 'RollingStd_4']]
)
sarima_result = sarima_model.fit(disp=False)
sarima_forecast_result = sarima_result.get_forecast(
    steps=test_weeks,
    exog=test_df[['RollingMean_4', 'RollingStd_4']]
)
sarima_forecast = sarima_forecast_result.predicted_mean
sarima_ci = sarima_forecast_result.conf_int()


def evaluate_model(true, pred, model_name, train_true, train_pred):
    print(f"\n {model_name} Performance:")
    print(f"Train RMSE: {np.sqrt(mean_squared_error(train_true, train_pred)):.2f} | Train R²: {r2_score(train_true, train_pred):.4f}")
    print(f"Val   RMSE: {np.sqrt(mean_squared_error(true, pred)):.2f} | Val   R²: {r2_score(true, pred):.4f}")

evaluate_model(test_series, arima_forecast, "ARIMA", train_series, arima_result.fittedvalues)

evaluate_model(test_series, sarima_forecast, "SARIMA", train_series, sarima_result.fittedvalues)


plt.figure(figsize=(14, 6))
plt.plot(df_agg.index, df_agg['Weekly_Sales'], label='Historical Sales', color='black')

plt.plot(test_dates, arima_forecast, label='ARIMA Forecast', linestyle='--', marker='o', color='blue')
plt.fill_between(test_dates, arima_ci.iloc[:, 0], arima_ci.iloc[:, 1], color='blue', alpha=0.2)

plt.plot(test_dates, sarima_forecast, label='SARIMA Forecast', linestyle='--', marker='x', color='green')
plt.fill_between(test_dates, sarima_ci.iloc[:, 0], sarima_ci.iloc[:, 1], color='green', alpha=0.2)

plt.axvline(test_dates[0], color='gray', linestyle=':', label='Train/Test Split')
plt.title("ARIMA vs SARIMA Forecasts with Rolling Features")
plt.xlabel("Date")
plt.ylabel("Weekly Sales")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
