from eda import open_file


def split_data(data):
    train = data[:'2017']
    test = data['2017':]
    return train, test


def baseline(data):
    data['naive'] = data['PJME_MW'].shift(1)
    data['seasonal_naive_24'] = data['PJME_MW'].shift(24)
    data['seasonal_naive_168'] = data['PJME_MW'].shift(168)


def main():
    data = open_file("data/PJME_hourly.csv")
    train, test = split_data(data)
    baseline(data)

if __name__ == "__main__":
    main()