from datetime import datetime
from pathlib import Path

from google.cloud import storage

from config import get_settings

DS = datetime.now().strftime("%Y-%m-%d")
_settings = get_settings()
LOCAL_FILE = _settings.out_dir / f"labelled_data_{DS}.csv"
GCS_BLOB_PATH = f"data/labelled_data_{DS}.csv"


def upload_to_gcs_bucket() -> None:
    """Upload today's labelled output file to the configured GCS bucket."""
    client = storage.Client()
    bucket = client.bucket(_settings.gcs_bucket_name)
    blob = bucket.blob(GCS_BLOB_PATH)
    blob.upload_from_filename(str(LOCAL_FILE))
    print(f"Uploaded {LOCAL_FILE} to gs://{_settings.gcs_bucket_name}/{GCS_BLOB_PATH}")


if __name__ == "__main__":
    upload_to_gcs_bucket()
