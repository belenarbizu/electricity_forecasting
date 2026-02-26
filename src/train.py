from eda import open_file
from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np
from statsmodels.tsa.statespace.sarimax import SARIMAX
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV


def prepare_data(data):
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


def split_data(data):
    train = data.loc['2014':'2016']
    X_train = train.drop(columns=['PJME_MW'])
    y_train = train['PJME_MW']
    test = data.loc['2017']
    X_test = test.drop(columns=['PJME_MW'])
    y_test = test['PJME_MW']
    return X_train, y_train, X_test, y_test


def baseline(data):
    data['seasonal_naive_24'] = data['PJME_MW'].shift(24)
    data['seasonal_naive_168'] = data['PJME_MW'].shift(168)


def baseline_eval(X_test, y_test):
    mae_naive_24 = mean_absolute_error(y_test, X_test['seasonal_naive_24'])
    rmse_naive_24 = np.sqrt(mean_squared_error(y_test, X_test['seasonal_naive_24']))

    mae_naive_168 = mean_absolute_error(y_test, X_test['seasonal_naive_168'])
    rmse_naive_168 = np.sqrt(mean_squared_error(y_test, X_test['seasonal_naive_168']))

    print(f"Seasonal Naive 24 MAE: {mae_naive_24:.2f}, RMSE: {rmse_naive_24:.2f}")
    print(f"Seasonal Naive 168 MAE: {mae_naive_168:.2f}, RMSE: {rmse_naive_168:.2f}")


def sarima_model(y_train, X_test, y_test):
    model = SARIMAX(y_train, order=(1, 0, 1), seasonal_order=(1, 1, 1, 24), enforce_stationarity=False, enforce_invertibility=False)
    results = model.fit(disp=False)
    print(results.summary())
    forecast = results.get_forecast(steps=len(X_test))
    forecast_values = forecast.predicted_mean
    forecast_values.index = X_test.index  # Align forecast index with test index
    mae_sarima = mean_absolute_error(y_test, forecast_values)
    rmse_sarima = np.sqrt(mean_squared_error(y_test, forecast_values))
    print(f"SARIMA MAE: {mae_sarima:.2f}, RMSE: {rmse_sarima:.2f}")


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

    for timestamp in X_test.index:
        row = X_test.loc[[timestamp]].copy()

        row['lag_1'] = history.iloc[-1]
        row['lag_24'] = history.iloc[-24]
        row['lag_168'] = history.iloc[-168]

        row['rolling_24'] = history.iloc[-24:].mean()
        row['rolling_168'] = history.iloc[-168:].mean()

        pred = best_model.predict(row)[0]
        predictions.append(pred)

        history = pd.concat([history, pd.Series(pred, index=[timestamp])])

    predictions = pd.Series(predictions, index=X_test.index)

    mae_gb = mean_absolute_error(y_test, predictions)
    rmse_gb = np.sqrt(mean_squared_error(y_test, predictions))

    print(f"Gradient Boosting MAE: {mae_gb:.2f}, RMSE: {rmse_gb:.2f}")

    # predictions = model.predict(X_test)
    # mae_gb = mean_absolute_error(y_test, predictions)
    # rmse_gb = np.sqrt(mean_squared_error(y_test, predictions))
    # print(f"Gradient Boosting MAE: {mae_gb:.2f}, RMSE: {rmse_gb:.2f}")


def main():
    data = open_file("data/PJME_hourly.csv")
    if data is None:
        return
    data.index = pd.to_datetime(data.index)
    data = data[~data.index.duplicated(keep='first')]
    data = data.sort_index()
    data = data.asfreq('h')
    data = data.interpolate()
    prepare_data(data)
    baseline(data)
    X_train, y_train, X_test, y_test = split_data(data)
    baseline_eval(X_test, y_test)
    X_train.drop(columns=['seasonal_naive_24', 'seasonal_naive_168'], inplace=True)
    X_test.drop(columns=['seasonal_naive_24', 'seasonal_naive_168'], inplace=True)
    gradient_boosting_model(X_train, y_train, X_test, y_test)
    X_train.drop(columns=['lag_1', 'lag_24', 'lag_168'], inplace=True)
    X_test.drop(columns=['lag_1', 'lag_24', 'lag_168'], inplace=True)
    #sarima_model(y_train, X_test, y_test)


if __name__ == "__main__":
    main()