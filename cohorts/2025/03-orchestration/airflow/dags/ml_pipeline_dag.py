from airflow.decorators import dag, task
from scripts.download import download_to_s3
from scripts.preprocess_data import run_data_prep
from scripts.train import train_and_register_model

S3_BUCKET = "s3-bucket"
S3_KEY = "data/input_data.parquet"
DATA_URL = "https://example.com/data/input_data.parquet"
MLFLOW_TRACKING_URI = "http://mlflow-tracking-server:5000"

@dag(schedule=None, catchup=False, tags=["ml_pipeline"])
def ml_pipeline_dag():

    @task
    def task_download():
        return download_to_s3(DATA_URL, S3_BUCKET, S3_KEY)

    @task
    def task_preprocess(s3_uri: str):
        return run_data_prep(s3_uri)

    @task
    def task_train(preprocessed_path: str):
        return train_and_register_model(preprocessed_path, MLFLOW_TRACKING_URI)

    s3_uri = task_download()
    preprocessed_path = task_preprocess(s3_uri)
    task_train(preprocessed_path)

ml_pipeline_dag = ml_pipeline_dag()
