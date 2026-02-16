import pandas as pd
import matplotlib.pyplot as plt
import os


def open_file(file_path):
    try:
        df = pd.read_csv(file_path, parse_dates=['Datetime'], index_col='Datetime')
        df.sort_index(inplace=True)
        return df
    except FileNotFoundError:
        print(f"File not found: {file_path}")
        return None
    except Exception as e:
        print(f"An error occurred: {e}")
        return None


def plot_data(df):
    plt.figure(figsize=(12, 6))
    plt.plot(df.index, df['PJME_MW'], label='Hourly Demand (MW)')
    plt.title('Hourly Electricity Demand')
    plt.xlabel('Date')
    plt.ylabel('Demand (MW)')
    plt.legend()
    plt.grid()
    plt.savefig('plots/hourly_electricity_demand.png')


def plot_a_week(df):
    plt.figure(figsize=(12, 6))
    week_data = df['2008-01-01':'2008-01-07']
    plt.plot(week_data.index, week_data['PJME_MW'], label='Hourly Demand (MW)')
    plt.title('Hourly Electricity Demand (First Week of January 2008)')
    plt.xlabel('Date')
    plt.ylabel('Demand (MW)')
    plt.legend()
    plt.grid()
    plt.savefig('plots/hourly_electricity_demand_week.png')


def plot_a_year(df):
    plt.figure(figsize=(12, 6))
    year_data = df['2008-01-01':'2008-12-31']
    plt.plot(year_data.index, year_data['PJME_MW'], label='Hourly Demand (MW)')
    plt.title('Hourly Electricity Demand (Year 2008)')
    plt.xlabel('Date')
    plt.ylabel('Demand (MW)')
    plt.legend()
    plt.grid()
    plt.savefig('plots/hourly_electricity_demand_year.png')


def hourly_heatmap(df):
    df_hourly = df.resample('h').mean()
    df_hourly['Hour'] = df_hourly.index.hour
    df_hourly['DayOfWeek'] = df_hourly.index.dayofweek
    heatmap_data = df_hourly.pivot_table(index='DayOfWeek', columns='Hour', values='PJME_MW')
    plt.figure(figsize=(12, 6))
    plt.imshow(heatmap_data, aspect='auto', cmap='viridis')
    plt.colorbar(label='Hourly Electricity Demand (MW)')
    plt.title('Hourly Electricity Demand Heatmap')
    plt.xlabel('Hour of Day')
    plt.ylabel('Day of Week')
    plt.xticks(range(24))
    plt.yticks(range(7), ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'])
    plt.savefig('plots/hourly_electricity_demand_heatmap.png')


def main():
    df = open_file("data/PJME_hourly.csv")
    if df is None:
        return
    print(df.head())
    print(df.info())
    if not os.path.exists('plots'):
        os.makedirs('plots')
    plot_data(df)
    plot_a_week(df)
    plot_a_year(df)
    hourly_heatmap(df)


if __name__ == "__main__":
    main()