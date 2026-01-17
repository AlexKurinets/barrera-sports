# This script loads all historical prediction files (pred_raw_*.csv) from AWS S3 bucket 'barrera-sports/predictions/',
# concatenates them into a single DataFrame, merges with the current 'bet_data/bet_data_new.csv' to obtain actual outcomes ('W/L'),
# filters to completed bets ('W' or 'L'), and computes the optimal threshold x (0 to 1, step 0.01) for Prob_W such that fading bets
# (taking the opposite side) where Prob_W < x maximizes total profit. Assumes flat 1 unit stake per fade, profit = (Decimal_Odds - 1)
# if bettor loses (Win=0), else -1. Uses same odds for fade approximation.

import json
import boto3
import pandas as pd
import numpy as np
import io
from botocore.exceptions import ClientError

if __name__ == "__main__":
    # Get secrets
    secrets_client = boto3.client('secretsmanager', region_name='us-east-2')
    secret_name = 'secrets_key'
    response = secrets_client.get_secret_value(SecretId=secret_name)
    secrets = json.loads(response['SecretString'])
    aws_access_key = secrets['aws_access_key_id']
    aws_secret_key = secrets['aws_secret_access_key']
    s3_bucket = secrets['bucket_name']

    s3 = boto3.client('s3', aws_access_key_id=aws_access_key, aws_secret_access_key=aws_secret_key)

    # List all pred_raw_*.csv files
    pred_files = []
    continuation_token = None
    while True:
        kwargs = {'Bucket': s3_bucket, 'Prefix': 'predictions/'}
        if continuation_token:
            kwargs['ContinuationToken'] = continuation_token
        response = s3.list_objects_v2(**kwargs)
        for obj in response.get('Contents', []):
            key = obj['Key']
            if 'pred_' in key and 'pred_raw_' not in key and key.endswith('.csv'):
                pred_files.append(key)
        if 'NextContinuationToken' not in response:
            break
        continuation_token = response['NextContinuationToken']

    if not pred_files:
        print("No pred_raw files found.")
        exit()

    # Download and concat all pred_raw
    dfs = []
    for key in pred_files:
        try:
            obj = s3.get_object(Bucket=s3_bucket, Key=key)
            df = pd.read_csv(io.BytesIO(obj['Body'].read()))
            dfs.append(df)
        except ClientError as e:
            print(f"Error downloading {key}: {e}")
    if not dfs:
        print("No data loaded.")
        exit()
    all_pred_df = pd.concat(dfs, ignore_index=True)
    all_pred_df = all_pred_df.drop_duplicates(subset=['Account', 'Bet Number'], keep='last')
    all_pred_df['Account'] = all_pred_df['Account'].astype(str).str.strip()
    all_pred_df['Bet Number'] = pd.to_numeric(all_pred_df['Bet Number'], errors='coerce')

    # Download bet_data_new.csv
    data_key = 'bet_data/bet_data_new.csv'
    try:
        obj = s3.get_object(Bucket=s3_bucket, Key=data_key)
        bet_data_df = pd.read_csv(io.BytesIO(obj['Body'].read()), encoding='latin1')
        bet_data_df['Date'] = pd.to_datetime(bet_data_df['Date'])
        bet_data_df['Account'] = bet_data_df['Account'].astype(str).str.strip()
        bet_data_df['Bet Number'] = pd.to_numeric(bet_data_df['Bet Number'].astype(str).str.replace(',', ''), errors='coerce')
    except ClientError as e:
        print(f"Error downloading {data_key}: {e}")
        exit()

    # Merge to get W/L
    merged_df = all_pred_df.merge(bet_data_df[['Account', 'Bet Number', 'W/L']], on=['Account', 'Bet Number'], how='left')
    merged_df = merged_df[merged_df['W/L'].isin(['W', 'L'])]
    merged_df['Win'] = np.where(merged_df['W/L'] == 'W', 1, 0)
    merged_df['Decimal Odds'] = pd.to_numeric(merged_df['Decimal Odds'], errors='coerce')

    # Drop rows with invalid odds or NaN
    merged_df = merged_df.dropna(subset=['Prob_W', 'Win', 'Decimal Odds'])
    merged_df = merged_df[merged_df['Decimal Odds'] > 1.0]

    if merged_df.empty:
        print("No valid data for optimization.")
        exit()

    # Optimize threshold
    opt_profit = -np.inf
    opt_x = 0.0
    opt_roi = 0.0
    opt_num_bets = 0
    for x in np.arange(0.0, 1.01, 0.01):
        faded = merged_df[merged_df['Prob_W'] < x]
        if faded.empty:
            continue
        num_bets = len(faded)
        fade_profit = np.where(faded['Win'] == 0, faded['Decimal Odds'] - 1, -1)
        total_profit = fade_profit.sum()
        total_staked = num_bets  # flat 1 unit per bet
        roi = total_profit / total_staked if total_staked > 0 else 0
        if total_profit > opt_profit:
            opt_profit = total_profit
            opt_x = x
            opt_roi = roi
            opt_num_bets = num_bets

    print(f"Optimal x: {opt_x:.2f}, Profit: {opt_profit:.2f}, ROI: {opt_roi:.2f}, Num Bets: {opt_num_bets}")