import os
import joblib
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path
from meteostat import Stations, Daily

DS = datetime.now().strftime("%Y-%m-%d")
BASE_DIR = Path(__file__).parent
MODEL_PATH = BASE_DIR / "isolation_forest.pkl"
DATA_DIR = Path(os.getenv("DATA_DIR", str(BASE_DIR / "data")))
OUT_DIR = Path(os.getenv("OUT_DIR", str(BASE_DIR / "data" / "scheduled_runs")))
FILE_PATH = DATA_DIR / "labelled_weather_data.csv"
OUT_PATH = OUT_DIR / f"labelled_data_{DS}.csv"

# boulder, co weather station coordinates
STATION_LAT = 40.014986
STATION_LON = -105.270546


def main() -> None:
    df = read_data()
    start, end = get_data_start_end(df)
    new_data = get_new_data(start, end)
    labelled_data = label_new_data(new_data)
    write_file(df, labelled_data)
    print_confirmation(df, labelled_data, start)


def read_data() -> pd.DataFrame:
    return pd.read_csv(FILE_PATH)


def get_data_start_end(dataframe: pd.DataFrame) -> tuple[datetime, datetime]:
    now = datetime.now()
    if len(dataframe) == 0:
        end = datetime(now.year, now.month, now.day) - timedelta(days=1)
        return end, end
    start = pd.to_datetime(dataframe["date"].iloc[-1]) + timedelta(days=1)
    end = datetime(now.year, now.month, now.day) + timedelta(days=1)
    return start, end


def get_new_data(start: datetime, end: datetime) -> pd.DataFrame:
    stations = Stations().nearby(STATION_LAT, STATION_LON)
    station_id = stations.fetch(1).reset_index()["id"][0]
    data = Daily(station_id, start, end).fetch()
    data = data[["tavg"]]
    data.index.name = "date"
    data.reset_index(inplace=True)
    data["date"] = data["date"].dt.strftime("%Y-%m-%d")
    return data


def label_new_data(dataframe: pd.DataFrame) -> pd.DataFrame:
    data = dataframe.copy()
    isolation_forest = joblib.load(MODEL_PATH)
    test_data: np.ndarray = np.array(data["tavg"]).reshape(-1, 1)
    labels: np.ndarray = isolation_forest.predict(test_data)
    data["anomaly_score"] = labels
    data["anomaly"] = np.where(data["anomaly_score"] == -1, True, False)
    return data


def write_file(dataframe: pd.DataFrame, labelled_data: pd.DataFrame) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.concat([dataframe, labelled_data], ignore_index=True)
    df.to_csv(OUT_PATH, index=False)


def print_confirmation(
    dataframe: pd.DataFrame, labelled_data: pd.DataFrame, start_date: datetime
) -> None:
    df = pd.concat([dataframe, labelled_data], ignore_index=True)
    total_anomalies = len(df.loc[df["anomaly"] == True])
    df["date"] = pd.to_datetime(df["date"])
    recent_anomalies = len(
        df.loc[(df["date"] > start_date.strftime("%Y-%m-%d")) & (df["anomaly"] == True)]
    )
    print(f"There are {total_anomalies} days with anomalous weather logged")
    print(f"New anomalies since {start_date.strftime('%Y-%m-%d')}: {recent_anomalies}")


if __name__ == "__main__":
    main()
