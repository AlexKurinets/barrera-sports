import json
import boto3
from botocore.exceptions import ClientError
import pandas as pd
import requests
from io import StringIO
import time
import logging
import os
from datetime import datetime

os.makedirs('logs', exist_ok=True)
logging.basicConfig(
    filename=f"C:/users/Administrator/barrera-sports/logs/data_ingest_to_aws_{datetime.now().strftime('%Y%m%d%H%M%S')}.log",
    level=logging.INFO
)

# Get secrets
secrets_client = boto3.client('secretsmanager', region_name='us-east-2')
secret_name = 'secrets_key'
response = secrets_client.get_secret_value(SecretId=secret_name)
secrets = json.loads(response['SecretString'])
aws_access_key = secrets['aws_access_key_id']
aws_secret_key = secrets['aws_secret_access_key']
s3_bucket = secrets['bucket_name']

while True:
    # Download Google Sheet as CSV
    sheet_id = '1wM00QhayythUnOizO7hIzv4hDHQutmTS'
    gid = '1467560424'
    export_url = f'https://docs.google.com/spreadsheets/d/{sheet_id}/export?format=csv&gid={gid}'
    response = requests.get(export_url)
    csv_content = response.content.decode('utf-8')
    df_new = pd.read_csv(StringIO(csv_content))
    df_new['Date'] = pd.to_datetime(df_new['Date']).dt.strftime("%Y-%m-%d")

    # S3 setup
    s3 = boto3.client('s3', aws_access_key_id=aws_access_key, aws_secret_access_key=aws_secret_key)
    s3_key = 'bet_data/bet_data.csv'

    # Get existing data
    try:
        obj = s3.get_object(Bucket=s3_bucket, Key=s3_key)
        df_existing = pd.read_csv(obj['Body'])
        max_date = df_existing['Date'].max()
        df_append = df_new[df_new['Date'] > max_date]
    except ClientError as e:
        if e.response['Error']['Code'] == 'NoSuchKey':
            df_append = df_new
            df_existing = pd.DataFrame(columns=df_new.columns)
        else:
            raise

    # Upload new records
    if not df_append.empty:
        df_updated = pd.concat([df_existing, df_append], ignore_index=True)
        csv_buffer = StringIO()
        df_updated.to_csv(csv_buffer, index=False)
        s3.put_object(Bucket=s3_bucket, Key=s3_key, Body=csv_buffer.getvalue())
        logging.info(f"{datetime.now().strftime('%Y-%m-%d')} Loaded {len(df_append)} new records.")
    else:
        print("No new records.")
        logging.info(f"{datetime.now().strftime('%Y-%m-%d')} No new records.")
    time.sleep(600)  # Sleep for 10 minutes
