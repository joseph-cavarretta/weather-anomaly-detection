from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    data_dir: Path = Field(default=Path("src/data"), description="Input data directory")
    out_dir: Path = Field(
        default=Path("src/data/scheduled_runs"), description="Output directory for labelled files"
    )
    gcs_bucket_name: str = Field(default="", description="GCS bucket for archival uploads")
    weather_model_dir: Path = Field(
        default=Path("~/projects/weather-anomaly-detection"),
        description="Project root used by the Airflow DAG",
    )


def get_settings() -> Settings:
    """Return a Settings instance loaded from environment and .env file."""
    return Settings()
