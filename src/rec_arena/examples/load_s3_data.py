#!/usr/bin/env python3
"""
Load data from S3 bucket for RecArena experiments.
Uses AWS credentials for example-account account.
"""

import boto3
import pandas as pd
import s3fs
from pathlib import Path
import os

def load_s3_data():
    """Load data from example-bucket S3 bucket."""
    
    # S3 configuration
    bucket_name = 'example-bucket'
    key = 'recarena/ml_100k/leave_one_out/test.parquet'
    s3_path = f's3://{bucket_name}/{key}'
    
    try:
        print(f"Loading: {s3_path}")
        
        # Method 1: Using s3fs with AWS profile
        fs = s3fs.S3FileSystem(profile='example-account')
        df = pd.read_parquet(s3_path, filesystem=fs)
        
        print(f"Successfully loaded {key}")
        print(f"Shape: {df.shape}")
        print(f"Columns: {list(df.columns)}")
        print(f"Data types:\n{df.dtypes}")
        print(f"\nFirst 10 rows:\n{df.head(10)}")
        print(f"\nBasic statistics:\n{df.describe()}")
        
        return df
            
    except Exception as e:
        print(f"Error with s3fs method: {e}")
        
        # Fallback: Method 2 - Direct pandas with s3 URL
        try:
            print("Trying direct pandas method...")
            # Set AWS profile in environment
            os.environ['AWS_PROFILE'] = 'example-account'
            df = pd.read_parquet(s3_path)
            
            print(f"Successfully loaded with direct method")
            print(f"Shape: {df.shape}")
            return df
            
        except Exception as e2:
            print(f"Error with direct method: {e2}")
            print("Make sure you're authenticated with AWS: aws sso login")
            return None

if __name__ == "__main__":
    data = load_s3_data()
