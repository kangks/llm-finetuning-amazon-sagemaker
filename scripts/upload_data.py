import os
import boto3
import sagemaker
from pathlib import Path

def upload_training_data():
    # Initialize session
    session = boto3.Session()
    sagemaker_session = sagemaker.Session(boto_session=session)
    
    # Get default bucket
    bucket = sagemaker_session.default_bucket()
    
    # Path to training data
    data_dir = Path('training_data')
    if not data_dir.exists():
        raise ValueError(f"Training data directory not found: {data_dir}")
    
    # Upload all JSON files in the training_data directory
    s3_client = session.client('s3')
    s3_prefix = 'training_data'
    
    for file_path in data_dir.glob('*.json'):
        s3_key = f'{s3_prefix}/{file_path.name}'
        print(f"Uploading {file_path} to s3://{bucket}/{s3_key}")
        s3_client.upload_file(str(file_path), bucket, s3_key)
    
    print(f"Training data uploaded to s3://{bucket}/{s3_prefix}/")
    return f's3://{bucket}/{s3_prefix}/'

if __name__ == '__main__':
    upload_training_data()