from datetime import datetime, timedelta

from airflow import DAG
from airflow.operators.bash import BashOperator

from config import get_settings

_settings = get_settings()
PROJECT_DIR = _settings.weather_model_dir

args = {
    "owner": "joe-cavarretta",
    "depend_on_past": False,
    "start_date": datetime(2023, 1, 1),
    "email_on_failure": False,
    "email_on_retry": False,
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
}

with DAG(
    dag_id="weather_model",
    default_args=args,
    description="Runs anomaly detection model on Boulder, CO recent weather",
    schedule_interval="0 12 * * 1",
) as dag:
    run_model = BashOperator(
        task_id="run_model",
        bash_command=(
            f"docker run --rm "
            f"--volume {PROJECT_DIR}/src/data:/data "
            f"--volume {PROJECT_DIR}/src/data/scheduled_runs:/output "
            f"weather-model"
        ),
    )
    upload_to_gcs = BashOperator(
        task_id="upload_to_gcs",
        bash_command=f"python3 {PROJECT_DIR}/scripts/upload_to_gcs.py",
    )

run_model >> upload_to_gcs
