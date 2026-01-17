##### optimize_fade_follow_threshold.py
# This script loads all historical prediction files (pred_raw_*.csv) from AWS S3 bucket 'barrera-sports/predictions/',
# concatenates them into a single DataFrame, merges with the current 'bet_data/bet_data_new.csv' to obtain actual outcomes ('W/L'),
# filters to completed bets ('W' or 'L'), and computes the optimal threshold x (0 to 1, step 0.01) for Prob_W such that fading bets
# (taking the opposite side) where Prob_W < x maximizes total profit. Assumes flat 1 unit stake per fade, profit = (Decimal_Odds - 1)
# if bettor loses (Win=0), else -1. Uses same odds for fade approximation.

import json
import boto3
import pandas as pd
import plotly.graph_objects as go
import numpy as np
import io
from botocore.exceptions import ClientError
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode',
                        choices=['fade', 'follow'],
                        default='fade',
                        help='Strategy mode: fade or follow'
                        )
    parser.add_argument('--start_date',
                        type=str,
                        default="2025-12-15",
                        help='Start date for analysis (YYYY-MM-DD)'
                        )
    args = parser.parse_args()
    mode = args.mode

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
    all_pred_df = pd.concat([df[['Date', 'Account', 'Prob_W', 'Team', 'Sport', 'League',
                                 'Unit Size', 'Wager', 'Odds', 'Type', 'Spread', 'Spread Type',
                                 'Total', 'Total Type']] for df in dfs], ignore_index=True)
    all_pred_df['Account'] = all_pred_df['Account'].astype(str).str.strip()

    # Download bet_data_new.csv
    data_key = 'bet_data/bet_data_new.csv'
    try:
        obj = s3.get_object(Bucket=s3_bucket, Key=data_key)
        bet_data_df = pd.read_csv(io.BytesIO(obj['Body'].read()), encoding='latin1')
        # bet_data_df['Date'] = pd.to_datetime(bet_data_df['Date'])
        bet_data_df['Account'] = bet_data_df['Account'].astype(str).str.strip()
    except ClientError as e:
        print(f"Error downloading {data_key}: {e}")
        exit()

    # Merge to get W/L
    merged_df = all_pred_df.merge(
        bet_data_df[['Account', 'Date', 'Type', 'Team', 'Sport', 'League', 'Odds', 'W/L', 'Decimal Odds']],
        on=['Account', 'Date', 'Type', 'Team', 'Sport', 'League', 'Odds'], how='left')
    merged_df = merged_df[merged_df['W/L'].isin(['W', 'L'])]
    merged_df['Win'] = np.where(merged_df['W/L'] == 'W', 1, 0)
    # merged_df['Decimal Odds'] = pd.to_numeric(merged_df['Decimal Odds_x'].combine_first(merged_df['Decimal Odds_y']),
    #                                           errors='coerce')
    # merged_df = merged_df.drop(columns=['Decimal Odds_x', 'Decimal Odds_y'], errors='ignore')

    # Drop rows with invalid odds or NaN
    merged_df = merged_df.dropna(subset=['Prob_W', 'Win', 'Decimal Odds'])
    merged_df = merged_df[merged_df['Decimal Odds'] > 1.0]

    if args.start_date:
        merged_df = merged_df[merged_df['Date'] >= args.start_date]

    if merged_df.empty:
        print("No valid data for optimization.")
        exit()

    start_date = merged_df['Date'].min()
    end_date = merged_df['Date'].max()

    results = []
    opt_profit = -np.inf
    opt_x = 0.0
    opt_roi = 0.0
    opt_num_bets = 0
    opt_pnl = 0.0
    for x in np.arange(0.01, 1.01, 0.01):
        if mode == 'fade':
            strategy_df = merged_df[merged_df['Prob_W'] < x]
        else:  # follow
            strategy_df = merged_df[merged_df['Prob_W'] >= x]
        if strategy_df.empty:
            continue
        num_bets = len(strategy_df)
        if mode == 'fade':
            profit_arr = np.where(strategy_df['Win'] == 0, strategy_df['Decimal Odds'] - 1, -1)
        else:  # follow
            profit_arr = np.where(strategy_df['Win'] == 1, strategy_df['Decimal Odds'] - 1, -1)
        total_profit = profit_arr.sum()
        total_staked = num_bets  # flat 1 unit per bet
        roi = total_profit / total_staked if total_staked > 0 else 0
        pnl = total_profit * 100  # $100 per bet
        strategy_name = 'Fade' if mode == 'fade' else 'Follow'
        print(
            f"{strategy_name} Threshold {x * 100:.0f}%: Profit: {total_profit:.2f}, ROI: {roi:.2f}, Num Bets: {num_bets}, $pnl: {pnl:.2f}")
        results.append({'threshold': x * 100, 'pnl': pnl, 'num_bets': num_bets})
        if total_profit > opt_profit:
            opt_profit = total_profit
            opt_x = x
            opt_roi = roi
            opt_num_bets = num_bets
            opt_pnl = pnl
    print(
        f"\nOptimal {strategy_name} x: {opt_x:.2f}, Profit: {opt_profit:.2f}, ROI: {opt_roi:.2f}, Num Bets: {opt_num_bets}, $pnl: {opt_pnl:.2f}")

    # Plot
    df_plot = pd.DataFrame(results)
    df_plot.to_csv(f"output/{mode}_threshold_{start_date}_{end_date}.csv")
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df_plot['threshold'], y=df_plot['pnl'], mode='lines+markers', name='$pnl', yaxis='y'))
    fig.add_trace(
        go.Scatter(x=df_plot['threshold'], y=df_plot['num_bets'], mode='lines+markers', name='Num Bets', yaxis='y2'))
    fig.update_layout(
        title=f'{strategy_name} Threshold Analysis {start_date} to {end_date}',
        xaxis_title='Threshold %',
        yaxis_title='$pnl',
        yaxis2=dict(title='Num Bets', overlaying='y', side='right'),
        template='plotly_white'
    )
    fig.write_html(f'output/plotting/{mode}_threshold_{start_date}_{end_date}_plot.html')
    print(f"Plot saved to {mode}_threshold_{start_date}_{end_date}_plot.html")