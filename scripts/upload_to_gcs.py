"""
Created on Tue Jan 17 20:17:12 2022
@author: joseph
"""
import os
from datetime import datetime
from google.cloud import storage

DS = datetime.now().strftime("%Y-%m-%d")
BUCKET_PATH = f'gs://joe-test-bucket-373803/data/labelled_data_{DS}.csv'
FILEPATH = f'/home/yosyp/projects/weather_model/src/data/scheduled_runs/labelled_data_{DS}.csv'
GCP_CREDS_PATH = '/home/yosyp/config/gcp_creds.json'
os.environ['GOOGLE_APPLICATION_CREDNETIALS'] = GCP_CREDS_PATH

def upload_to_gcs_bucket():
    client = storage.Client()
    bucket = client.get_bucket(BUCKET_PATH)
    blob = bucket.blob(FILEPATH)
    blob.upload_from_filename(filename=FILEPATH)


if __name__ == '__main__':
    upload_to_gcs_bucket()