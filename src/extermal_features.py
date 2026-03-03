from meteostat import Station, hourly
from datetime import datetime
import pandas as pd


base_temp = 18.0


def get_temperature_data(id, start_date, end_date):
    # Create a Point for the location
    location = Station(id=id)

    # Fetch hourly temperature data
    data = hourly(location, start_date, end_date)
    data = data.fetch()

    data = data[['temp']]  # Keep only the temperature column
    data.index = pd.to_datetime(data.index)  # Ensure the index is datetime
    data = data.asfreq('h')  # Resample to hourly frequency, filling missing values with NaN

    # Calculate HDD and CDD
    data['HDD'] = (base_temp - data['temp']).clip(lower=0)
    data['CDD'] = (data['temp'] - base_temp).clip(lower=0)

    return data


if __name__ == "__main__":
    # Example usage
    id = '72408' # Philadelphia Airport
    start_date = datetime(2014, 1, 1)
    end_date = datetime(2017, 12, 31)

    data = get_temperature_data(id, start_date, end_date)
    print(data.head())