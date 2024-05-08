"""
Created on Fri Apr  8 17:27:05 2022
@author: joseph.cavarretta
"""
import pandas as pd
import numpy as np
from statsmodels.tsa.seasonal import STL
from sklearn.ensemble import IsolationForest
import joblib


def main():
    df = load_data_and_format('/src/data/weather_data_historical.csv')
    print('Training isolation forest model...')
    train_isolation_forest(df)
    print('Model training complete.')


def load_data_and_format(path):
    """Reads original weather data csv and returns processed dataframe.
    Args:
        path: the path to the original weather data csv
    """
    raw_df = pd.read_csv(path)
    # change dt_iso to datetime
    # original format: YYYY-MM-DD 00:00:00 +0000 UTC
    raw_df['dt_iso'] = raw_df['dt_iso'].str[:-10]
    raw_df['dt_iso'] = pd.to_datetime(raw_df['dt_iso'], format='%Y-%m-%d %H:%M:%S')
    raw_df.sort_values('dt_iso', inplace=True)
    # set datetime index for downsampling
    raw_df.set_index('dt_iso', inplace=True)
    # create copy of raw_df and downsample hourly -> daily
    df = raw_df.resample('D').agg({'temp':'mean'}).rename(columns={'temp':'mean_temp'})

    # add additional columns for max and min temp of each day
    df['max_temp'] = raw_df['temp'].resample('D').max()
    df['min_temp'] = raw_df['temp'].resample('D').min()

    stl = STL(df['mean_temp'])
    res = stl.fit()
    df['LOESS_residuals'] = res.resid

    df.to_csv('/src/data/processed_weather_data_historical.csv', index=False)
    return df


def train_isolation_forest(df):
    """Trains an isolation forest on the original weather data.
    Args:
        df - processed dataframe
    Returns:
        df with data labelled as anomolous or not
    """
    isolation_forest = IsolationForest(n_estimators=100, n_jobs=-1, contamination=0.05)
    isolation_forest.fit(df['LOESS_residuals'].to_numpy().reshape(-1,1))

    df['iso_forest_anomaly'] = isolation_forest.fit_predict(df['LOESS_residuals'].to_numpy().reshape(-1,1))

    joblib.dump(isolation_forest, "isolation_forest.pkl")

    return


if __name__ == '__main__':
    main()
