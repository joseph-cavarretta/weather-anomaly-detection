import os
from datetime import datetime
from pathlib import Path
from google.cloud import storage

DS = datetime.now().strftime("%Y-%m-%d")
BUCKET_NAME = os.environ["GCS_BUCKET_NAME"]
LOCAL_FILE = Path(os.getenv("OUT_DIR", "src/data/scheduled_runs")) / f"labelled_data_{DS}.csv"
GCS_BLOB_PATH = f"data/labelled_data_{DS}.csv"


def upload_to_gcs_bucket() -> None:
    client = storage.Client()
    bucket = client.bucket(BUCKET_NAME)
    blob = bucket.blob(GCS_BLOB_PATH)
    blob.upload_from_filename(str(LOCAL_FILE))
    print(f"Uploaded {LOCAL_FILE} to gs://{BUCKET_NAME}/{GCS_BLOB_PATH}")


if __name__ == "__main__":
    upload_to_gcs_bucket()
