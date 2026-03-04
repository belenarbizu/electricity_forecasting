from eda import open_file
from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np
from statsmodels.tsa.statespace.sarimax import SARIMAX
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
import argparse
from xgboost import XGBRegressor, plot_importance
from extermal_features import get_temperature_data
from datetime import datetime
import matplotlib.pyplot as plt
import mlflow


def prepare_data(data):
    data.index = pd.to_datetime(data.index)
    data = data[~data.index.duplicated(keep='first')]
    data = data.sort_index()
    data = data.asfreq('h')
    data = data.interpolate()

    # This way 23 and 0 are close to each other, which is important for time series data
    data['hour_sin'] = np.sin(2 * np.pi * data.index.hour / 24)
    # Cosine transformation to capture the cyclical nature of hours in a day (06 is not the same as 18)
    data['hour_cos'] = np.cos(2 * np.pi * data.index.hour / 24)

    data['day_of_week'] = data.index.dayofweek
    data['month'] = data.index.month
    data['year'] = data.index.year
    data['day_of_year'] = data.index.dayofyear
    data['day_of_month'] = data.index.day
    data['quarter'] = data.index.quarter
    data['is_weekend'] = data.index.dayofweek // 5

    data['lag_1'] = data['PJME_MW'].shift(1)
    data['lag_24'] = data['PJME_MW'].shift(24)
    data['lag_168'] = data['PJME_MW'].shift(168)

    data['rolling_24'] = data['PJME_MW'].shift(1).rolling(24).mean()
    data['rolling_168'] = data['PJME_MW'].shift(1).rolling(168).mean()

    data.dropna(inplace=True)
    data = merge_data(data)
    return data


def merge_data(data):
    # Merge temperature data with the main dataset
    start_date = datetime(2014, 1, 1)
    end_date = datetime(2017, 12, 31)
    temp_data = get_temperature_data('72408', start_date, end_date)
    data = data.merge(temp_data, left_index=True, right_index=True, how='left')

    data['trend'] = np.arange(len(data))

    return data


def split_data(data):
    train = data.loc['2014':'2016']
    train = train.dropna()
    X_train = train.drop(columns=['PJME_MW'])
    y_train = train['PJME_MW']

    test = data.loc['2017']
    test = test.dropna()
    X_test = test.drop(columns=['PJME_MW'])
    y_test = test['PJME_MW']

    return X_train, y_train, X_test, y_test


def baseline_eval(X_test, y_test):
    mae_naive_24 = mean_absolute_error(y_test, X_test['lag_24'])
    rmse_naive_24 = np.sqrt(mean_squared_error(y_test, X_test['lag_24']))

    mae_naive_168 = mean_absolute_error(y_test, X_test['lag_168'])
    rmse_naive_168 = np.sqrt(mean_squared_error(y_test, X_test['lag_168']))

    print(f"Seasonal Naive 24 MAE: {mae_naive_24:.2f}, RMSE: {rmse_naive_24:.2f}")
    print(f"Seasonal Naive 168 MAE: {mae_naive_168:.2f}, RMSE: {rmse_naive_168:.2f}")

    return mae_naive_24, rmse_naive_24, mae_naive_168, rmse_naive_168


def sarima_model(y_train, X_test, y_test):
    model = SARIMAX(y_train, order=(1, 0, 1), seasonal_order=(1, 1, 1, 24), enforce_stationarity=False, enforce_invertibility=False)
    results = model.fit(disp=False)
    
    forecast = results.get_forecast(steps=len(X_test))
    forecast_values = forecast.predicted_mean
    forecast_values.index = X_test.index  # Align forecast index with test index

    mae_sarima = mean_absolute_error(y_test, forecast_values)
    rmse_sarima = np.sqrt(mean_squared_error(y_test, forecast_values))

    print(f"SARIMA MAE: {mae_sarima:.2f}, RMSE: {rmse_sarima:.2f}")
    
    return model, mae_sarima, rmse_sarima


def gradient_boosting_model(X_train, y_train, X_test, y_test):
    param_grid = {
        'n_estimators': np.arange(50, 200, 20),
        'learning_rate': [0.01, 0.05, 0.1],
        'max_depth': [3, 5, 7, 10]
    }
    tscv = TimeSeriesSplit(n_splits=3)
    model = GridSearchCV(GradientBoostingRegressor(random_state=42), param_grid, cv=tscv, n_jobs=-1)
    model.fit(X_train, y_train)

    best_model = model.best_estimator_

    history = y_train.copy()
    predictions = []

    # We need to predict one step at a time to update the lag features without leaking future information
    for timestamp in X_test.index:
        row = X_test.loc[[timestamp]].copy()

        # These lag features are predictions from the previous steps
        row['lag_1'] = history.iloc[-1]
        row['lag_24'] = history.iloc[-24]
        row['lag_168'] = history.iloc[-168]

        row['rolling_24'] = history.iloc[-24:].mean()
        row['rolling_168'] = history.iloc[-168:].mean()

        pred = best_model.predict(row)[0]
        predictions.append(pred)

        # Update the history with the new prediction for the next iteration
        history = pd.concat([history, pd.Series(pred, index=[timestamp])])

    predictions = pd.Series(predictions, index=X_test.index)

    mae_gb = mean_absolute_error(y_test, predictions)
    rmse_gb = np.sqrt(mean_squared_error(y_test, predictions))

    print(f"Gradient Boosting MAE: {mae_gb:.2f}, RMSE: {rmse_gb:.2f}")

    return best_model, mae_gb, rmse_gb


def xgboost_model(X_train, y_train, X_test, y_test):
    model = XGBRegressor(
        n_estimators=600,
        learning_rate=0.03,
        max_depth=8,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_lambda=1.0,
        reg_alpha=0.5,
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_train, y_train)

    history = y_train.copy()
    predictions = []

    for timestamp in X_test.index:
        row = X_test.loc[[timestamp]].copy()

        row['lag_1'] = history.iloc[-1]
        row['lag_24'] = history.iloc[-24]
        row['lag_168'] = history.iloc[-168]

        row['rolling_24'] = history.iloc[-24:].mean()
        row['rolling_168'] = history.iloc[-168:].mean()

        pred = model.predict(row)[0]
        predictions.append(pred)

        history = pd.concat([history, pd.Series(pred, index=[timestamp])])

    predictions = pd.Series(predictions, index=X_test.index)

    mae_xgb = mean_absolute_error(y_test, predictions)
    rmse_xgb = np.sqrt(mean_squared_error(y_test, predictions))

    print(f"XGBoost MAE: {mae_xgb:.2f}, RMSE: {rmse_xgb:.2f}")

    plt.figure(figsize=(10, 8))
    plot_importance(model, title='XGBoost Feature Importance', importance_type='gain')
    plt.tight_layout()
    plt.savefig('plots/xgb_feature_importance.png')
    plt.close()

    plt.figure(figsize=(12, 6))
    plt.plot(y_test.index, y_test.values, label='Actual', color='blue')
    plt.plot(predictions.index, predictions.values, label='Predicted', color='orange', alpha=0.7)
    plt.title('XGBoost Predictions vs Actual')
    plt.xlabel('Date')
    plt.ylabel('Electricity Demand (MW)')
    plt.legend()
    plt.tight_layout()
    plt.savefig('plots/xgb_predictions.png')
    plt.close()

    error_df = pd.DataFrame({
        'error': y_test.values - predictions,
        'hour': y_test.index.hour
    })
    hourly_error = error_df.groupby('hour')['error'].mean()
    plt.figure(figsize=(10, 6))
    hourly_error.plot()
    plt.title('Average Prediction Error by Hour of Day')
    plt.xlabel('Hour of Day')
    plt.ylabel('Average Prediction Error (MW)')
    plt.xticks(range(24))
    plt.tight_layout()
    plt.savefig('plots/hourly_error.png')
    plt.close()

    return model, mae_xgb, rmse_xgb


def mlflow_logging_model(model, model_name, mae, rmse):
    mlflow.set_experiment("Electricity Demand Forecasting")
    
    with mlflow.start_run(run_name=model_name):
        if model is not None:
            mlflow.sklearn.log_model(model, "model")

        if model is not None and hasattr(model, 'get_params'):
            mlflow.log_params(model.get_params())
        elif model_name == 'SARIMA Model':
            mlflow.log_param("order_p", 1)
            mlflow.log_param("order_d", 0)
            mlflow.log_param("order_q", 1)

            mlflow.log_param("seasonal_P", 1)
            mlflow.log_param("seasonal_D", 1)
            mlflow.log_param("seasonal_Q", 1)
            mlflow.log_param("seasonal_s", 24)
        else:
            mlflow.log_params({})

        mlflow.log_metrics({"MAE": mae, "RMSE": rmse})
        


def main():
    argument_parser = argparse.ArgumentParser(description="Train models for electricity demand forecasting")
    argument_parser.add_argument('--gb', action='store_true', help="Train Gradient Boosting model")
    argument_parser.add_argument('--sarima', action='store_true', help="Train SARIMA model")
    argument_parser.add_argument('--xgb', action='store_true', help="Train XGBoost model")
    args = argument_parser.parse_args()

    data = open_file("data/PJME_hourly.csv")
    if data is None:
        return

    data = prepare_data(data)

    X_train, y_train, X_test, y_test = split_data(data)

    mae_24, rmse_24, mae_168, rmse_168 = baseline_eval(X_test, y_test)
    mlflow_logging_model(None, 'Baseline Model 24', mae_24, rmse_24)
    mlflow_logging_model(None, 'Baseline Model 168', mae_168, rmse_168)

    if args.gb:
        model, mae, rmse = gradient_boosting_model(X_train, y_train, X_test, y_test)
        mlflow_logging_model(model, 'Gradient Boosting Model', mae, rmse)
    if args.sarima:
        X_train.drop(columns=['lag_1', 'lag_24', 'lag_168'], inplace=True)
        X_test.drop(columns=['lag_1', 'lag_24', 'lag_168'], inplace=True)
        model, mae, rmse = sarima_model(y_train, X_test, y_test)
        mlflow_logging_model(model, 'SARIMA Model', mae, rmse)
    if args.xgb:
        model, mae, rmse = xgboost_model(X_train, y_train, X_test, y_test)
        mlflow_logging_model(model, 'XGBoost Model', mae, rmse)



if __name__ == "__main__":
    main()