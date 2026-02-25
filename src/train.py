from eda import open_file
from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np


def split_data(data):
    train = data[:'2017']
    test = data['2017':]
    return train, test


def baseline(data):
    data['seasonal_naive_24'] = data['PJME_MW'].shift(24)
    data['seasonal_naive_168'] = data['PJME_MW'].shift(168)


def baseline_eval(test):
    mae_naive_24 = mean_absolute_error(test['PJME_MW'], test['seasonal_naive_24'])
    rmse_naive_24 = np.sqrt(mean_squared_error(test['PJME_MW'], test['seasonal_naive_24']))

    mae_naive_168 = mean_absolute_error(test['PJME_MW'], test['seasonal_naive_168'])
    rmse_naive_168 = np.sqrt(mean_squared_error(test['PJME_MW'], test['seasonal_naive_168']))

    print(f"Seasonal Naive 24 MAE: {mae_naive_24:.2f}, RMSE: {rmse_naive_24:.2f}")
    print(f"Seasonal Naive 168 MAE: {mae_naive_168:.2f}, RMSE: {rmse_naive_168:.2f}")


def main():
    data = open_file("data/PJME_hourly.csv")
    baseline(data)
    train, test = split_data(data)
    baseline_eval(test)


if __name__ == "__main__":
    main()