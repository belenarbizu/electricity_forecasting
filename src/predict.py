import joblib
import argparse
import os
import pandas as pd
import numpy as np

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, '..', 'models', 'xgboost_model.pkl')
INFO_PATH = os.path.join(BASE_DIR, '..', 'models', 'info.pkl')

model = joblib.load(MODEL_PATH)
info = joblib.load(INFO_PATH)

MAX_HOURS_AHEAD = 24


def calculate_lags(datetime, temperature):
    history = info['history'].copy()
    last_datetime = pd.to_datetime(info['last_date'])
    trend = info['trend']

    try:
        datetime = pd.to_datetime(datetime)
    except ValueError:
        raise

    if datetime <= last_datetime:
        raise ValueError('Date must be after 31/12/2016 23:00')
    
    hours_ahead = int((datetime - last_datetime).total_seconds() // 3600)
    if hours_ahead > MAX_HOURS_AHEAD:
        raise ValueError(f"You can only predict up to {MAX_HOURS_AHEAD} hours ahead")

    current_time = last_datetime
    prediction = None

    for _ in range(hours_ahead):
        current_time += pd.Timedelta(hours=1)
        trend += 1

        features = build_features(current_time, temperature, history, trend)
        prediction = model.predict(features)[0]
        history.append(prediction)
    
    return prediction


def build_features(datetime, temperature, history, trend):
    lag_1 = history[-1]
    lag_24 = history[-24]
    lag_168 = history[-168]

    rolling_24 = np.mean(history[-24:])
    rolling_168 = np.mean(history[-168:])

    input_data = pd.DataFrame({
        'day_of_week': [datetime.dayofweek],
        'month': [datetime.month],
        'year': [datetime.year],
        'hour_sin': [np.sin(2 * np.pi * datetime.hour / 24)],
        'hour_cos': [np.cos(2 * np.pi * datetime.hour / 24)],
        'day_of_year': [datetime.dayofyear],
        'day_of_month': [datetime.day],
        'quarter': [datetime.quarter],
        'is_weekend': [datetime.dayofweek // 5],
        'temp': [temperature],
        'HDD': [max(0, 18 - temperature)],
        'CDD': [max(0, temperature - 18)],
        'trend': [trend],
        'lag_1': [lag_1],
        'lag_24': [lag_24],
        'lag_168': [lag_168],
        'rolling_24': [rolling_24],
        'rolling_168': [rolling_168]
    })

    input_data = input_data[model.feature_names_in_]
    return input_data


def main():
    argument_parser = argparse.ArgumentParser(description="Load and use a trained model for prediction")
    argument_parser.add_argument("--date", type=str, required=True, help="Date for prediction")
    argument_parser.add_argument('--temp', type=float, required=True, help='Temperature for prediction')
    args = argument_parser.parse_args()

    if model is None:
        return
    
    prediction = calculate_lags(args.date, args.temp)
    print(f"Prediction: {prediction}")

if __name__ == "__main__":
    main()