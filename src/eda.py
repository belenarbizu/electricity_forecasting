import pandas as pd
import matplotlib.pyplot as plt
import os
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf


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
    plt.savefig('plots/hourly_demand.png')


def plot_a_week(df):
    plt.figure(figsize=(12, 6))
    week_data = df['2008-01-01':'2008-01-07']
    plt.plot(week_data.index, week_data['PJME_MW'], label='Hourly Demand (MW)')
    plt.title('Hourly Electricity Demand (First Week of January 2008)')
    plt.xlabel('Date')
    plt.ylabel('Demand (MW)')
    plt.legend()
    plt.grid()
    plt.savefig('plots/hourly_demand_week.png')


def plot_a_year(df):
    plt.figure(figsize=(12, 6))
    year_data = df['2008-01-01':'2008-12-31']
    plt.plot(year_data.index, year_data['PJME_MW'], label='Hourly Demand (MW)')
    plt.title('Hourly Electricity Demand (Year 2008)')
    plt.xlabel('Date')
    plt.ylabel('Demand (MW)')
    plt.legend()
    plt.grid()
    plt.savefig('plots/hourly_demand_year.png')


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
    plt.savefig('plots/hourly_demand_heatmap.png')


def autocorrelation(df):
    fig, axes = plt.subplots(2, 1, figsize=(12, 10))
    plot_acf(df['PJME_MW'], lags=200, ax=axes[0])
    plot_pacf(df['PJME_MW'], lags=200, ax=axes[1])
    axes[0].set_title('Autocorrelation of Hourly Electricity Demand')
    axes[1].set_title('Partial Autocorrelation of Hourly Electricity Demand')
    plt.tight_layout()
    plt.savefig('plots/hourly_demand_autocorrelation.png')


def distribution(df):
    plt.figure(figsize=(12, 6))
    plt.hist(df['PJME_MW'], bins=50, color='blue', alpha=0.7)
    plt.title('Distribution of Hourly Electricity Demand')
    plt.xlabel('Demand (MW)')
    plt.ylabel('Frequency')
    plt.grid()
    plt.savefig('plots/hourly_demand_distribution.png')


def outliers(df):
    Q1 = df['PJME_MW'].quantile(0.25)
    Q3 = df['PJME_MW'].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    outliers = df[(df['PJME_MW'] < lower_bound) | (df['PJME_MW'] > upper_bound)]
    plt.figure(figsize=(12, 6))
    plt.scatter(outliers.index, outliers['PJME_MW'], color='red', label='Outliers')
    plt.plot(df.index, df['PJME_MW'], label='Hourly Demand (MW)', alpha=0.5)
    plt.title('Outliers in Hourly Electricity Demand')
    plt.xlabel('Date')
    plt.ylabel('Demand (MW)')
    plt.legend()
    plt.grid()
    plt.savefig('plots/hourly_demand_outliers.png')


def annual_trend(df):
    df['Year'] = df.index.year
    annual_demand = df.groupby('Year')['PJME_MW'].mean()
    plt.figure(figsize=(12, 6))
    plt.plot(annual_demand.index, annual_demand.values, marker='o', label='Annual Average Demand (MW)')
    plt.title('Annual Average Electricity Demand')
    plt.xlabel('Year')
    plt.ylabel('Average Demand (MW)')
    plt.legend()
    plt.grid()
    plt.savefig('plots/annual_demand_trend.png')


def daily_trend(df):
    df['Hour'] = df.index.hour
    hourly_demand = df.groupby('Hour')['PJME_MW'].mean()
    plt.figure(figsize=(12, 6))
    plt.plot(hourly_demand.index, hourly_demand.values, marker='o', label='Average Demand by Hour (MW)')
    plt.title('Average Electricity Demand by Hour of Day')
    plt.xlabel('Hour of Day')
    plt.ylabel('Average Demand (MW)')
    plt.legend()
    plt.grid()
    plt.savefig('plots/hourly_demand_trend.png')


def weekly_trend(df):
    df['DayOfWeek'] = df.index.dayofweek
    daily_demand = df.groupby('DayOfWeek')['PJME_MW'].mean()
    plt.figure(figsize=(12, 6))
    plt.plot(daily_demand.index, daily_demand.values, marker='o', label='Average Demand by Day of Week (MW)')
    plt.title('Average Electricity Demand by Day of Week')
    plt.xlabel('Day of Week')
    plt.ylabel('Average Demand (MW)')
    plt.xticks(range(7), ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'])
    plt.legend()
    plt.grid()
    plt.savefig('plots/weekly_demand_trend.png')


def main():
    df = open_file("data/PJME_hourly.csv")
    if df is None:
        return

    if not os.path.exists('plots'):
        os.makedirs('plots')

    autocorrelation(df)
    distribution(df)
    outliers(df)

    annual_trend(df)
    daily_trend(df)
    weekly_trend(df)
    
    plot_data(df)
    plot_a_week(df)
    plot_a_year(df)
    hourly_heatmap(df)


if __name__ == "__main__":
    main()