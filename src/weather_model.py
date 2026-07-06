from datetime import datetime, timedelta
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from meteostat import Daily, Stations

from config import get_settings

BASE_DIR = Path(__file__).parent
MODEL_PATH = BASE_DIR / "isolation_forest.pkl"

STATION_LAT = 40.014986
STATION_LON = -105.270546


def main() -> None:
    """Run inference: fetch new weather data, label it, and append to history."""
    settings = get_settings()
    ds = datetime.now().strftime("%Y-%m-%d")
    file_path = settings.data_dir / "labelled_weather_data.csv"
    out_path = settings.out_dir / f"labelled_data_{ds}.csv"

    df = read_data(file_path)
    start, end = get_data_start_end(df)
    new_data = get_new_data(start, end)
    labelled_data = label_new_data(new_data)
    write_file(df, labelled_data, settings.out_dir, out_path)
    print_confirmation(df, labelled_data, start)


def read_data(path: Path) -> pd.DataFrame:
    """Load the existing labelled weather history."""
    return pd.read_csv(path)


def get_data_start_end(dataframe: pd.DataFrame) -> tuple[datetime, datetime]:
    """Return the start and end dates for the next data fetch.

    Args:
        dataframe: Existing labelled history. If empty, fetches yesterday only.

    Returns:
        Tuple of (start, end) datetimes for the meteostat query.
    """
    now = datetime.now()
    if len(dataframe) == 0:
        end = datetime(now.year, now.month, now.day) - timedelta(days=1)
        return end, end
    start = pd.to_datetime(dataframe["date"].iloc[-1]) + timedelta(days=1)
    # meteostat station data lags ~24h; end must be tomorrow to capture today
    end = datetime(now.year, now.month, now.day) + timedelta(days=1)
    return start, end


def get_new_data(start: datetime, end: datetime) -> pd.DataFrame:
    """Fetch daily average temperature from the nearest meteostat station.

    Args:
        start: First date to fetch.
        end: Last date to fetch (exclusive of today due to station lag).

    Returns:
        DataFrame with columns: date, tavg.
    """
    stations = Stations().nearby(STATION_LAT, STATION_LON)
    station_id = stations.fetch(1).reset_index()["id"][0]
    data = Daily(station_id, start, end).fetch()
    data = data[["tavg"]]
    data.index.name = "date"
    data.reset_index(inplace=True)
    data["date"] = data["date"].dt.strftime("%Y-%m-%d")
    return data


def label_new_data(dataframe: pd.DataFrame) -> pd.DataFrame:
    """Apply the trained Isolation Forest to label new observations.

    Args:
        dataframe: DataFrame with tavg column from get_new_data.

    Returns:
        Input DataFrame with anomaly_score and anomaly columns added.
    """
    data = dataframe.copy()
    isolation_forest = joblib.load(MODEL_PATH)
    test_data: np.ndarray = np.array(data["tavg"]).reshape(-1, 1)
    labels: np.ndarray = isolation_forest.predict(test_data)
    data["anomaly_score"] = labels
    data["anomaly"] = np.where(data["anomaly_score"] == -1, True, False)
    return data


def write_file(
    dataframe: pd.DataFrame,
    labelled_data: pd.DataFrame,
    out_dir: Path,
    out_path: Path,
) -> None:
    """Append newly labelled data to history and write to output path."""
    out_dir.mkdir(parents=True, exist_ok=True)
    df = pd.concat([dataframe, labelled_data], ignore_index=True)
    df.to_csv(out_path, index=False)


def print_confirmation(
    dataframe: pd.DataFrame,
    labelled_data: pd.DataFrame,
    start_date: datetime,
) -> None:
    """Print a summary of total and new anomalies detected."""
    df = pd.concat([dataframe, labelled_data], ignore_index=True)
    total_anomalies = len(df.loc[df["anomaly"]])
    df["date"] = pd.to_datetime(df["date"])
    recent_anomalies = len(
        df.loc[(df["date"] > start_date.strftime("%Y-%m-%d")) & df["anomaly"]]
    )
    print(f"There are {total_anomalies} days with anomalous weather logged")
    print(f"New anomalies since {start_date.strftime('%Y-%m-%d')}: {recent_anomalies}")


if __name__ == "__main__":
    main()
