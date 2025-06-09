import requests
import boto3
import os
import pandas as pd

RAW_DATA_PATH = "/opt/airflow/data/raw"

def download_raw_data():
    import requests
    import os

    parquet_urls = [
        "https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_2023-03.parquet"
    ]
    os.makedirs(RAW_DATA_PATH, exist_ok=True)
    for url in parquet_urls:
        print(f"Downloading {url}...")
        filename = os.path.join(RAW_DATA_PATH, os.path.basename(url))
        response = requests.get(url)
        if response.status_code != 200:
            print(f"Error downloading {url}: {response.status_code}")
            print(response.text)
            raise Exception(f"Failed to download {url}: {response.status_code}")
        if os.path.exists(filename):
            print(f"File {filename} already exists. Skipping download.")
            continue
        response.raise_for_status()
        with open(filename, "wb") as f:
            print(f"Saving to {filename}...")
            f.write(response.content)
    
        print(f"Downloaded {filename}")