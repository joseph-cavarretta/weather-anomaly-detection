from datetime import datetime

from google.cloud import storage

from config import get_settings


def upload_to_gcs_bucket() -> None:
    """Upload today's labelled output file to the configured GCS bucket."""
    settings = get_settings()
    ds = datetime.now().strftime("%Y-%m-%d")
    local_file = settings.out_dir / f"labelled_data_{ds}.csv"
    blob_path = f"data/labelled_data_{ds}.csv"

    client = storage.Client()
    bucket = client.bucket(settings.gcs_bucket_name)
    blob = bucket.blob(blob_path)
    blob.upload_from_filename(str(local_file))
    print(f"Uploaded {local_file} to gs://{settings.gcs_bucket_name}/{blob_path}")


if __name__ == "__main__":
    upload_to_gcs_bucket()
