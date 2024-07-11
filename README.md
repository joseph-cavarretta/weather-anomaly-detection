## Unsupervised Anomaly Detection with Weather Data

The purpose of this project is to label historical weather data, specifical mean daily temp, as anomolous or not,
and use the trained model to label new data points. The historical data used is specifically for the town of Boulder, CO and is sourced from
[Open Weather](https://home.openweathermap.org/). It contains hourly weather data from 1970 to 2020.

### This project contains the following files:
[weather_EDA.ipynb](https://github.com/joseph-cavarretta/weather-anomaly-detection/blob/main/src/weather_eda.ipynb)

Includes various data prep, visuals, and model selection. Compares Isolation Forest model against using traditional anomaly detection methods
with Standard Deviation and Inter-Quartile Range.

[train_model.py](https://github.com/joseph-cavarretta/weather-anomaly-detection/blob/main/src/train_model.py)

Trains an Isolation Forest model on the [50 year historical data](https://github.com/joseph-cavarretta/weather-anomaly-detection/blob/main/src/data/weather_data_historical.csv.gz). Hourly data is resampled into daily averages.
Isolation Forest _contamination_ is set to 0.05, indicating the percentage of outliers we expect in this data set. Given my meteorological experience (none)
this threshold seemed to work well for this data set, however someone with specific domain knowledge may be able to advise better on this.

Seasonal Trend Decomposition is applied to the data using LOESS to extract trend, seasonality, and residual components. The Isolation Forest
is then trained on the residual component.

Running this file loads the original historical dataset and re-trains the model on it, saving the model to isolation_forest.pkl for use in
weather_model.py. Processed historical data is saved as processed_weather_data_historical.csv.

[weather_model.py](https://github.com/joseph-cavarretta/weather-anomaly-detection/blob/main/src/weather_model.py)

Uses the Meteostat API to pull weather data from the closest weather station to the historical weather data that I could find.
Checking this data against common weather outlets (such as Accuweather), shows some variation. Because of this, the new data that is pulled is not appended to
the original historical data for now.

Data is pulled starting from the last day in the [labelled_weather_data_csv](https://github.com/joseph-cavarretta/weather-anomaly-detection/blob/main/src/data/labelled_weather_data.csv.gz) file
and ending yesterday. Using the Isolation Forest this data is labelled as anomalous or not and appended to the labelled weather data csv file.
