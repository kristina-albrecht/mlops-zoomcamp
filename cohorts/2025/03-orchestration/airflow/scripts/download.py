import requests
import boto3
from urllib.parse import urlparse
import os
import pandas as pd

def download_to_s3(parquet_url: str, bucket: str, s3_key: str) -> str:
    # Download the parquet file
    response = requests.get(parquet_url, stream=True)
    response.raise_for_status()
    
    # Save to a temporary file
    local_filename = os.path.basename(urlparse(parquet_url).path)
    with open(local_filename, 'wb') as f:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)

    df = pd.read_parquet(local_filename)
    
    # Upload to S3
    s3 = boto3.client('s3')
    s3.upload_file(local_filename, bucket, s3_key)
    
    # Remove local file
    os.remove(local_filename)
    
    # Return S3 URL
    s3_url = f"s3://{bucket}/{s3_key}"
    return s3_url