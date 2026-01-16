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

s3_key_new = 'bet_data/bet_data_new.csv'
s3_key_old = 'bet_data/bet_data_old.csv'

while True:
    # Download Google Sheet as CSV
    sheet_id = '1wM00QhayythUnOizO7hIzv4hDHQutmTS'
    gid = '1467560424'
    export_url = f'https://docs.google.com/spreadsheets/d/{sheet_id}/export?format=csv&gid={gid}'
    response = requests.get(export_url)
    csv_content = response.content.decode('utf-8')
    df_new = pd.read_csv(StringIO(csv_content))
    df_new['Date'] = pd.to_datetime(df_new['Date']).dt.strftime("%Y-%m-%d")
    df_new['Account'] = df_new['Account'].astype(str).str.strip()
    df_new['Bet Number'] = pd.to_numeric(df_new['Bet Number'].astype(str).str.replace(',', ''), errors='coerce')
    df_new = df_new.drop_duplicates(subset=['Account', 'Bet Number'], keep='last')
    df_new = df_new.fillna('')

    # S3 setup
    s3 = boto3.client('s3', aws_access_key_id=aws_access_key, aws_secret_access_key=aws_secret_key)

    # Get existing data
    try:
        obj = s3.get_object(Bucket=s3_bucket, Key=s3_key_new)
        df_existing = pd.read_csv(obj['Body'])
        df_existing['Account'] = df_existing['Account'].astype(str).str.strip()
        df_existing['Bet Number'] = pd.to_numeric(df_existing['Bet Number'].astype(str).str.replace(',', ''),
                                                  errors='coerce')
        df_existing = df_existing.drop_duplicates(subset=['Account', 'Bet Number'], keep='last')
        df_existing = df_existing.fillna('')
        key_columns = ['Account', 'Bet Number']
        df_new_indexed = df_new.set_index(key_columns)
        df_existing_indexed = df_existing.set_index(key_columns)
        new_keys = df_new_indexed.index.difference(df_existing_indexed.index)
        common_keys = df_new_indexed.index.intersection(df_existing_indexed.index)
        updated_rows = 0
        updated_mask = None
        if not common_keys.empty:
            df_common_old = df_existing_indexed.loc[common_keys].copy()
            df_common_new = df_new_indexed.loc[common_keys].copy()
            df_common_old = df_common_old.apply(lambda col: col.str.strip() if col.dtype == 'object' else col)
            df_common_new = df_common_new.apply(lambda col: col.str.strip() if col.dtype == 'object' else col)
            is_equal = (df_common_old == df_common_new) | (df_common_old.isnull() & df_common_new.isnull())
            updated_mask = ~is_equal.all(axis=1)
            updated_rows = updated_mask.sum()
            if updated_rows > 0:
                logging.info(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} Updated {updated_rows} existing records.")
        updated_keys = common_keys[updated_mask] if updated_mask is not None else pd.MultiIndex.from_arrays(
            [[] for _ in key_columns])
        df_updated_indexed = df_existing_indexed.copy()
        if not updated_keys.empty:
            df_updated_indexed.loc[updated_keys] = df_new_indexed.loc[updated_keys]
        if not new_keys.empty:
            df_updated_indexed = pd.concat([df_updated_indexed, df_new_indexed.loc[new_keys]])
        df_updated = df_updated_indexed.reset_index()
        new_rows = len(new_keys) if not new_keys.empty else 0
    except ClientError as e:
        if e.response['Error']['Code'] == 'NoSuchKey':
            df_new['Account'] = df_new['Account'].astype(str).str.strip()
            df_new['Bet Number'] = pd.to_numeric(df_new['Bet Number'].astype(str).str.replace(',', ''), errors='coerce')
            df_updated = df_new.drop_duplicates(subset=['Account', 'Bet Number'], keep='last').fillna('')
            df_existing = pd.DataFrame(columns=df_new.columns)
            updated_rows = 0
            new_rows = len(df_updated)
        else:
            raise

    # Upload updated data if there are new records or updates
    if updated_rows > 0 or new_rows > 0:
        csv_buffer = StringIO()
        df_updated.to_csv(csv_buffer, index=False)
        try:
            s3.copy_object(Bucket=s3_bucket, CopySource=s3_bucket + '/' + s3_key_new, Key=s3_key_old)
            logging.info(
                f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} Copied previous bet_data_new.csv to bet_data_old.csv.")
        except ClientError as e:
            if e.response['Error']['Code'] == 'NoSuchKey':
                pass
            else:
                raise
        s3.put_object(Bucket=s3_bucket, Key=s3_key_new, Body=csv_buffer.getvalue())
        if new_rows > 0:
            logging.info(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} Loaded {new_rows} new records.")
        if updated_rows > 0:
            logging.info(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} Updated {updated_rows} existing records (may overlap with new if initial load).")
    else:
        logging.info(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} No new or updated records.")
    time.sleep(600)  # Sleep for 10 minutes
