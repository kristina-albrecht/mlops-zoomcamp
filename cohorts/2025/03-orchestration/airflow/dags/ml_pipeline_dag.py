from datetime import datetime
import os
import boto3
import logging

from airflow import DAG
from airflow.operators.python import PythonOperator

RAW_DATA_PATH = "/opt/airflow/data/raw"
DEST_PATH = "/opt/airflow/data/processed"
SCRIPT_PATH = "/opt/airflow/scripts/preprocess_data.py"
S3_BUCKET = "mlops-zoomcamp-taxi"
S3_PREFIX = "processed_data"
MLFLOW_TRACKING_URI = "http://localhost:5000"


def download_raw_data():
    import requests
    import os

    parquet_urls = [
        "https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_2023-03.parquet"
    ]
    os.makedirs(RAW_DATA_PATH, exist_ok=True)
    for url in parquet_urls:
        logging.info(f"Downloading {url}...")
        filename = os.path.join(RAW_DATA_PATH, os.path.basename(url))
        response = requests.get(url)
        if response.status_code != 200:
            logging.info(f"Error downloading {url}: {response.status_code}")
            logging.info(response.text)
            raise Exception(f"Failed to download {url}: {response.status_code}")
        if os.path.exists(filename):
            logging.info(f"File {filename} already exists. Skipping download.")
            continue
        response.raise_for_status()
        with open(filename, "wb") as f:
            logging.info(f"Saving to {filename}...")
            f.write(response.content)
    
        logging.info(f"Downloaded {filename}")

def run_preprocessing():
    import subprocess
    subprocess.run([
        "python", SCRIPT_PATH,
        "--raw_data_path", RAW_DATA_PATH,
        "--dest_path", DEST_PATH,
        "--dataset", "yellow"
    ], check=True)

def train_model():
    import subprocess
    subprocess.run([
        "python", "/opt/airflow/scripts/train.py",
        "--data_path", DEST_PATH
    ], check=True)

def register_model():
    import mlflow
    import os

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment("nyc-taxi-trip-duration")

    with mlflow.start_run():
        model_path = os.path.join(DEST_PATH, "model")
        mlflow.sklearn.log_model(model_path, "model")
        mlflow.log_artifact(os.path.join(DEST_PATH, "dv.pkl"), artifact_path="artifacts")
        mlflow.log_artifact(os.path.join(DEST_PATH, "train.pkl"), artifact_path="artifacts")
        mlflow.log_artifact(os.path.join(DEST_PATH, "val.pkl"), artifact_path="artifacts")
        mlflow.log_artifact(os.path.join(DEST_PATH, "test.pkl"), artifact_path="artifacts")

def upload_to_s3():
    session = boto3.Session()
    s3 = session.client("s3")

    for filename in ["dv.pkl", "train.pkl", "val.pkl", "test.pkl"]:
        local_path = os.path.join(DEST_PATH, filename)
        s3_key = f"{S3_PREFIX}/{filename}"

        s3.upload_file(local_path, S3_BUCKET, s3_key)
        print(f"Uploaded {filename} to s3://{S3_BUCKET}/{s3_key}")

with DAG(
    dag_id="ml_pipeline",
    start_date=datetime(2023, 1, 1),
    catchup=False,
    tags=["download", "preprocessing"],
) as dag:

    preprocess_task = PythonOperator(
        task_id="run_preprocessing",
        python_callable=run_preprocessing
    )

    train_task = PythonOperator(
        task_id="train_model",
        python_callable=train_model
    )

    register_task = PythonOperator(
        task_id="register_model",
        python_callable=register_model
    )

    preprocess_task >> train_task >> register_task