from datetime import timedelta, datetime
from airflow import DAG
from airflow.operators.bash import BashOperator
# docker operator performs equivalent of a run command
# from airflow.providers.docker.operators.docker import DockerOperator

args = {
    'owner': 'joe-cavarretta',
    'depend_on_past': False,
    'start_date': datetime(2023, 1, 1),
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5)
}

with DAG(
    dag_id='weather_model',
    default_args=args,
    description='Runs anomaly detection model on Boulder, CO recent weather',
    schedule_interval='0 12 * * 0' # run on mondays at noon
) as dag:
    run_model = BashOperator(
        task_id='run_model',
        bash_command='docker run --rm --volume ~/projects/weather_model/src/data:/data weather_model',
    )
    upload_to_gcs = BashOperator(
        task_id='upload_to_gcs',
        bash_command='python3 /home/yosyp/projects/weather_model/scripts/upload_to_gcs.py',
    )

run_model >> upload_to_gcs