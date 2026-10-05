import logging
from pathlib import Path
from typing import TYPE_CHECKING

import joblib
import pandas as pd
from sklearn.ensemble import IsolationForest
from statsmodels.tsa.seasonal import STL

if TYPE_CHECKING:
    import numpy as np

BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
MODEL_PATH = BASE_DIR / "isolation_forest.pkl"

logger = logging.getLogger(__name__)


def main() -> None:
    """Train the Isolation Forest on the historical data and save it."""
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    df = load_data_and_format(DATA_DIR / "weather_data_historical.csv.gz")
    logger.info("Training isolation forest model...")
    train_isolation_forest(df)
    logger.info("Model training complete.")


def load_data_and_format(path: Path) -> pd.DataFrame:
    """Load raw hourly weather data, resample to daily, and add STL residuals."""
    raw_df = pd.read_csv(path)
    raw_df["dt_iso"] = raw_df["dt_iso"].str[:-10]
    raw_df["dt_iso"] = pd.to_datetime(raw_df["dt_iso"], format="%Y-%m-%d %H:%M:%S")
    raw_df.sort_values("dt_iso", inplace=True)
    raw_df.set_index("dt_iso", inplace=True)

    df = (
        raw_df.resample("D").agg({"temp": "mean"}).rename(columns={"temp": "mean_temp"})
    )
    df["max_temp"] = raw_df["temp"].resample("D").max()
    df["min_temp"] = raw_df["temp"].resample("D").min()

    stl = STL(df["mean_temp"])
    res = stl.fit()
    df["LOESS_residuals"] = res.resid

    df.to_csv(DATA_DIR / "processed_weather_data_historical.csv", index=False)
    return df


def train_isolation_forest(df: pd.DataFrame) -> None:
    """Train an Isolation Forest on STL residuals and save the model."""
    isolation_forest = IsolationForest(n_estimators=100, n_jobs=-1, contamination=0.05)
    residuals: np.ndarray = df["LOESS_residuals"].to_numpy().reshape(-1, 1)
    isolation_forest.fit(residuals)
    df["iso_forest_anomaly"] = isolation_forest.fit_predict(residuals)
    joblib.dump(isolation_forest, MODEL_PATH)


if __name__ == "__main__":
    main()
