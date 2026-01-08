import json
import boto3
import pandas as pd
import numpy as np
import torch
from torch import nn
import os
import io
import joblib
import re
import requests
import time
from datetime import datetime, timezone
import logging
from botocore.exceptions import ClientError
import argparse

os.makedirs('logs', exist_ok=True)
logging.basicConfig(
    filename=f"C:/users/Administrator/barrera-sports/logs/predict_lstm_fade_{datetime.now().strftime('%Y%m%d%H%M%S')}.log",
    level=logging.INFO
)

class LSTMModel(nn.Module):
    def __init__(self, input_size_num, cat_cols, vocab_sizes, embed_dim, hidden_size, num_layers):
        super().__init__()
        self.cat_cols = cat_cols
        self.embeddings = nn.ModuleDict({col: nn.Embedding(vocab_sizes[col], embed_dim) for col in cat_cols})
        embed_size = len(cat_cols) * embed_dim
        self.lstm = nn.LSTM(input_size_num + embed_size, hidden_size, num_layers, batch_first=True, dropout=0.2)
        self.fc = nn.Linear(hidden_size, 1)
    def forward(self, x_num, x_cat):
        embeds = [self.embeddings[col](x_cat[:,:,i]) for i, col in enumerate(self.cat_cols)]
        embeds = torch.cat(embeds, dim=-1)
        x = torch.cat([x_num, embeds], dim=-1)
        _, (h, _) = self.lstm(x)
        out = self.fc(h[-1])
        return out

def parse_hyperparams(filename):  # Adjusted indices to account for 'fade_model_' prefix
    parts = filename.split('_')
    seq_len = int(parts[3])
    bs = int(parts[5])
    hs = int(parts[7])
    nl = int(parts[9])
    lr = float(parts[11])
    ep = int(parts[13])
    pat = int(parts[15])
    return seq_len, bs, hs, nl, lr, ep, pat

def preprocess_df(df, encoders, scaler):
    df = df.sort_values(['Account', 'Date'])
    categorical_cols = ['Type', 'Sport', 'League', 'Marketing', 'Spread Type', 'Total Type', 'ML Type',
                        'Odds Bracket', 'Wager Bracket', 'Team', 'Division']
    categorical_cols = [col for col in categorical_cols if col in df.columns]
    numerical_cols = ['Decimal Odds', 'Units Risked', 'Marketed Unit', 'Unit Size', 'Wager', 'Return %', 'Net $',
                      'Account Net to Date', 'Account Average Return % to Date', 'Bet Number', 'Spread', 'Total',
                      'Odds', 'Paid $', 'Account Net to Date for Month', 'Marked Unit Qualified',
                      'Spread Bracker Qualified', 'Sport Qualified', 'League Qualified', 'Marketing Qualified',
                      'Type Qualified', 'Spread Type Qualified', 'Total Type Qualified', 'ML Type Qualified',
                      'Date Qualified', 'Count Towards Account Net', 'Last Wager in Month']
    numerical_cols = [col for col in numerical_cols if col in df.columns]
    for col in numerical_cols:
        df[col] = pd.to_numeric(df[col].astype(str).str.replace(',', '').str.replace('$', '').str.replace('%', ''), errors='coerce')
    df[numerical_cols] = df[numerical_cols].fillna(0)
    predictor_cols = numerical_cols + categorical_cols + ['Date']
    untransformed_df = df[predictor_cols].copy()
    df['Win'] = np.where(df['W/L'] == 'W', 1.0, 0.0)
    df['Rolling_Win_Pct_5'] = df.groupby('Account')['Win'].transform(lambda x: x.rolling(5, min_periods=1).mean().shift(1))
    df['Rolling_Win_Pct_10'] = df.groupby('Account')['Win'].transform(lambda x: x.rolling(10, min_periods=1).mean().shift(1))
    df['Rolling_Win_Pct_3'] = df.groupby('Account')['Win'].transform(lambda x: x.rolling(3, min_periods=1).mean().shift(1))
    df['Rolling_Win_Pct_15'] = df.groupby('Account')['Win'].transform(lambda x: x.rolling(15, min_periods=1).mean().shift(1))
    df['Rolling_Win_Pct_30'] = df.groupby('Account')['Win'].transform(lambda x: x.rolling(30, min_periods=1).mean().shift(1))
    df['Rolling_Win_Pct_50'] = df.groupby('Account')['Win'].transform(lambda x: x.rolling(50, min_periods=1).mean().shift(1))
    df['Rolling_Return_5'] = df.groupby('Account')['Return %'].transform(lambda x: x.rolling(5, min_periods=1).mean().shift(1))
    df['Rolling_Return_10'] = df.groupby('Account')['Return %'].transform(lambda x: x.rolling(10, min_periods=1).mean().shift(1))
    df['Rolling_Return_3'] = df.groupby('Account')['Return %'].transform(lambda x: x.rolling(3, min_periods=1).mean().shift(1))
    df['Rolling_Return_15'] = df.groupby('Account')['Return %'].transform(lambda x: x.rolling(15, min_periods=1).mean().shift(1))
    df['Rolling_Return_30'] = df.groupby('Account')['Return %'].transform(lambda x: x.rolling(30, min_periods=1).mean().shift(1))
    df['Rolling_Return_50'] = df.groupby('Account')['Return %'].transform(lambda x: x.rolling(50, min_periods=1).mean().shift(1))
    df['Rolling_Net_5'] = df.groupby('Account')['Net $'].transform(lambda x: x.rolling(5, min_periods=1).sum().shift(1))
    df['Rolling_Net_10'] = df.groupby('Account')['Net $'].transform(lambda x: x.rolling(10, min_periods=1).sum().shift(1))
    df['Rolling_Net_20'] = df.groupby('Account')['Net $'].transform(lambda x: x.rolling(20, min_periods=1).sum().shift(1))
    df['Rolling_Net_30'] = df.groupby('Account')['Net $'].transform(lambda x: x.rolling(30, min_periods=1).sum().shift(1))
    df['Rolling_Odds_5'] = df.groupby('Account')['Decimal Odds'].transform(lambda x: x.rolling(5, min_periods=1).mean().shift(1))
    df['Rolling_Odds_10'] = df.groupby('Account')['Decimal Odds'].transform(lambda x: x.rolling(10, min_periods=1).mean().shift(1))
    df['Rolling_Return_Std_5'] = df.groupby('Account')['Return %'].transform(lambda x: x.rolling(5, min_periods=1).std().shift(1))
    df['Rolling_Return_Std_10'] = df.groupby('Account')['Return %'].transform(lambda x: x.rolling(10, min_periods=1).std().shift(1))
    df['EWM_Win_Pct'] = df.groupby('Account')['Win'].transform(lambda x: x.ewm(span=10, adjust=False).mean().shift(1))
    df['EWM_Return'] = df.groupby('Account')['Return %'].transform(
        lambda x: x.ewm(span=10, adjust=False).mean().shift(1))
    df['Lag1_Return'] = df.groupby('Account')['Return %'].shift(1)
    df['Lag1_Net'] = df.groupby('Account')['Net $'].shift(1)
    df['streak_group'] = df.groupby('Account')['Win'].transform(lambda x: (x.diff() != 0).cumsum())
    df['Streak'] = df.groupby(['Account', 'streak_group']).cumcount() + 1
    df['Streak'] = np.where(df['Win'] == 0, -df['Streak'], df['Streak'])
    df = df.drop('streak_group', axis=1)
    df['Days_Since_Last'] = df.groupby('Account')['Date'].diff().dt.days
    df['Rolling_Days_Between_5'] = df.groupby('Account')['Days_Since_Last'].transform(lambda x: x.rolling(5, min_periods=1).mean().shift(1))
    df['Day_of_Week'] = df['Date'].dt.dayofweek
    df['Is_Weekend'] = (df['Day_of_Week'] >= 5).astype(int)
    df['Month'] = df['Date'].dt.month
    df['Bet_Number_Log'] = np.log1p(df['Bet Number'])
    df['Wager_Log'] = np.log1p(df['Wager'])
    df['Days'] = (df['Date'] - df.groupby('Account')['Date'].transform('min')).dt.days
    new_cols = ['Rolling_Win_Pct_5', 'Rolling_Win_Pct_10', 'Rolling_Win_Pct_3', 'Rolling_Win_Pct_15',
                'Rolling_Win_Pct_30', 'Rolling_Win_Pct_50',
                'Rolling_Return_5', 'Rolling_Return_10', 'Rolling_Return_3', 'Rolling_Return_15', 'Rolling_Return_30',
                'Rolling_Return_50',
                'Rolling_Net_5', 'Rolling_Net_10', 'Rolling_Net_20', 'Rolling_Net_30',
                'Rolling_Odds_5', 'Rolling_Odds_10', 'Rolling_Return_Std_5', 'Rolling_Return_Std_10',
                'EWM_Win_Pct', 'EWM_Return', 'Lag1_Return', 'Lag1_Net', 'Streak', 'Days_Since_Last',
                'Rolling_Days_Between_5',
                'Day_of_Week', 'Is_Weekend', 'Month',
                'Bet_Number_Log', 'Wager_Log', 'Days']
    df[new_cols] = df[new_cols].fillna(0)
    numerical_cols += new_cols
    categorical_cols = ['Type', 'Sport', 'League', 'Marketing', 'Spread Type', 'Total Type', 'ML Type',
                        'Odds Bracket', 'Wager Bracket', 'Team', 'Division']
    categorical_cols = [col for col in categorical_cols if col in df.columns]
    for col in categorical_cols:
        mapping = dict(zip(encoders[col].classes_, range(len(encoders[col].classes_))))
        unseen = len(encoders[col].classes_)
        df[col] = df[col].astype(str).map(mapping).fillna(unseen).astype(int)
    df[numerical_cols] = scaler.transform(df[numerical_cols])
    return df, numerical_cols, categorical_cols, untransformed_df

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Predict LSTM Fade Script')
    parser.add_argument(
        '--filter_mode',
        type=str,
        choices=['new', 'date'],
        default='new',
        help='Filter mode: "new" for new rows only (default), "date" for most recent date'
    )
    args = parser.parse_args()
    filter_mode = args.filter_mode
    secrets_client = boto3.client('secretsmanager', region_name='us-east-2')
    secret_name = 'secrets_key'
    response = secrets_client.get_secret_value(SecretId=secret_name)
    secrets = json.loads(response['SecretString'])
    aws_access_key = secrets['aws_access_key_id']
    aws_secret_key = secrets['aws_secret_access_key']
    s3_bucket = secrets['bucket_name']
    telegram_bot_token = secrets['telegram_bot_token']
    telegram_chat_id = secrets['telegram_chat_id']
    s3 = boto3.client('s3', aws_access_key_id=aws_access_key, aws_secret_access_key=aws_secret_key)
    s3_key_new = 'bet_data/bet_data_new.csv'
    s3_key_old = 'bet_data/bet_data_old.csv'
    previous_last_modified = None
    previous_df = None
    while True:
        try:
            head = s3.head_object(Bucket=s3_bucket, Key=s3_key_new)
            current_last_modified = head['LastModified']
            if previous_last_modified is None or current_last_modified > previous_last_modified:
                logging.info(f"File updated at {current_last_modified}. Checking for new rows.")
                obj = s3.get_object(Bucket=s3_bucket, Key=s3_key_new)
                df = pd.read_csv(obj['Body'], encoding='latin1')
                df["Date"] = pd.to_datetime(df["Date"], format="%Y-%m-%d")
                if df.empty:
                    logging.info("bet_data_new.csv is empty. Skipping predictions.")
                    previous_last_modified = current_last_modified
                    continue
                df['Account'] = df['Account'].astype(str).str.strip()
                df['Bet Number'] = pd.to_numeric(df['Bet Number'].astype(str).str.replace(',', ''), errors='coerce')
                should_run = True
                if filter_mode == 'new':
                    try:
                        obj_old = s3.get_object(Bucket=s3_bucket, Key=s3_key_old)
                        df_old = pd.read_csv(obj_old['Body'], encoding='latin1')
                        df_old["Date"] = pd.to_datetime(df_old["Date"], format="%Y-%m-%d")
                        df_old['Account'] = df_old['Account'].astype(str).str.strip()
                        df_old['Bet Number'] = pd.to_numeric(df_old['Bet Number'].astype(str).str.replace(',', ''),
                                                             errors='coerce')
                    except ClientError as e:
                        if e.response['Error']['Code'] == 'NoSuchKey':
                            df_old = pd.DataFrame(columns=df.columns)
                        else:
                            raise
                    key_columns = ['Account', 'Bet Number']
                    df_indexed = df.set_index(key_columns)
                    df_old_indexed = df_old.set_index(key_columns)
                    new_keys = df_indexed.index.difference(df_old_indexed.index)
                    if new_keys.empty:
                        logging.info("No new rows detected via comparison. Skipping predictions.")
                        previous_last_modified = current_last_modified
                        should_run = False
                        continue
                    logging.info(f"{len(new_keys)} new rows detected via comparison. Running predictions.")
                    most_recent_bet_date = df_indexed.loc[new_keys, 'Date'].max()
                else:  # 'date'
                    logging.info("Running predictions for most recent date.")
                    most_recent_bet_date = df['Date'].max()
                most_recent_bet_date_str = most_recent_bet_date.strftime('%Y%m%d')
                if not should_run:
                    continue
                response = s3.list_objects_v2(Bucket=s3_bucket, Prefix='models/')
                files = [obj['Key'] for obj in response.get('Contents', []) if 'end_' in obj['Key']]
                dates = set()
                for f in files:
                    match = re.search(r'end_(\d{8})\.', f)
                    if match:
                        dates.add(match.group(1))
                sorted_dates = sorted(dates, reverse=True)
                if len(sorted_dates) < 2:
                    logging.info("Insufficient models available.")
                    continue
                most_recent_model_date_str = sorted_dates[0]
                model_files = [f for f in files if f'fade_model' in f and f'end_{most_recent_model_date_str}.pth' in f]
                joblib_files = [f for f in files if 'preprocessors' in f and f'end_{most_recent_model_date_str}.joblib' in f]
                if not model_files or not joblib_files:
                    logging.info("Model or preprocessor files not found for second latest date.")
                    continue
                model_key = model_files[0]
                joblib_key = joblib_files[0]
                obj = s3.get_object(Bucket=s3_bucket, Key=joblib_key)
                with io.BytesIO(obj['Body'].read()) as buffer:
                    preprocessors = joblib.load(buffer)
                scaler = preprocessors['scaler']
                encoders = preprocessors['encoders']
                obj = s3.get_object(Bucket=s3_bucket, Key=model_key)
                with io.BytesIO(obj['Body'].read()) as buffer:
                    state_dict = torch.load(buffer, map_location=torch.device('cpu'))
                filename = os.path.basename(model_key)
                sequence_length, _, hidden_size, num_layers, _, _, _ = parse_hyperparams(filename)
                if filter_mode == 'new':
                    df['is_new'] = df.set_index(key_columns).index.isin(new_keys)
                df, numerical_cols, categorical_cols, untransformed_df = preprocess_df(df, encoders, scaler)
                vocab_sizes = {col: len(encoders[col].classes_) + 1 for col in categorical_cols}
                embed_dim = 16
                input_size_num = len(numerical_cols)
                model = LSTMModel(input_size_num, categorical_cols, vocab_sizes, embed_dim, hidden_size, num_layers)
                for name, param in state_dict.items():
                    if 'embeddings' in name and 'weight' in name:
                        model_param = model.state_dict()[name]
                        if model_param.shape[0] > param.shape[0] and model_param.shape[1:] == param.shape[1:]:
                            model_param[:param.shape[0]].copy_(param)
                            model_param[param.shape[0]:] = model_param[:param.shape[0]].mean(dim=0)
                        elif model_param.shape == param.shape:
                            model_param.copy_(param)
                        else:
                            raise ValueError(f"Shape mismatch for {name}")
                    else:
                        model.state_dict()[name].copy_(param)
                model.eval()
                predictions = []
                groups = df.groupby('Account')
                for name, group in groups:
                    group = group.sort_values('Date').reset_index()
                    if filter_mode == 'new':
                        infer_rows = group[group['is_new']]
                    else:
                        infer_rows = group[group['Date'] == most_recent_bet_date]
                    for _, row in infer_rows.iterrows():
                        m = row.name
                        if m < sequence_length:
                            continue
                        seq_num = group.iloc[m - sequence_length:m][numerical_cols].values
                        seq_cat = group.iloc[m - sequence_length:m][categorical_cols].values
                        seq_num_t = torch.tensor(seq_num).float().unsqueeze(0)
                        seq_cat_t = torch.tensor(seq_cat).long().unsqueeze(0)
                        with torch.no_grad():
                            logit = model(seq_num_t, seq_cat_t).squeeze()
                            prob = torch.sigmoid(logit).item()
                            prob = round(prob, 3)
                        predictions.append(
                            {'Account': name, 'Prob_W': prob, **untransformed_df.loc[row['index']].to_dict()})
                pred_raw_df = pd.DataFrame(predictions)
                pred_raw_df["Date"] = pd.to_datetime(pred_raw_df["Date"]).dt.strftime("%Y-%m-%d")
                if filter_mode == 'new':
                    pred_raw_key = f'predictions/pred_raw_{most_recent_bet_date_str}.csv'
                    try:
                        obj = s3.get_object(Bucket=s3_bucket, Key=pred_raw_key)
                        existing_pred_raw_df = pd.read_csv(obj['Body'])
                        pred_raw_df = pd.concat([existing_pred_raw_df, pred_raw_df], ignore_index=True)
                        pred_raw_df = pred_raw_df.drop_duplicates()
                    except ClientError as e:
                        if e.response['Error']['Code'] != 'NoSuchKey':
                            raise
                columns = ['Date', 'Account', 'Prob_W'] + [col for col in pred_raw_df.columns if
                                                           col not in ['Date', 'Account', 'Prob_W']]
                pred_raw_df = pred_raw_df[columns]
                pred_raw_df = pred_raw_df.sort_values('Prob_W', ascending=True)
                csv_buffer = io.StringIO()
                pred_raw_df.to_csv(csv_buffer, index=False)
                s3.put_object(Bucket=s3_bucket, Key=f'predictions/pred_raw_{most_recent_bet_date_str}.csv',
                              Body=csv_buffer.getvalue())
                pred_df_columns = ['Date', 'Account', 'Prob_W', 'Team',
                                   'Sport', 'League',
                                   'Unit Size', 'Wager',
                                   'Odds', 'Type',
                                   'Spread', 'Spread Type',
                                   'Total', 'Total Type']
                pred_df = pred_raw_df[pred_df_columns]
                csv_buffer = io.StringIO()
                pred_df.to_csv(csv_buffer, index=False)
                s3.put_object(Bucket=s3_bucket, Key=f'predictions/pred_{most_recent_bet_date_str}.csv',
                              Body=csv_buffer.getvalue())
                csv_buffer.seek(0)
                url = f"https://api.telegram.org/bot{telegram_bot_token}/sendDocument"
                files = {'document': (f'pred_{most_recent_bet_date_str}.csv', csv_buffer.getvalue().encode('utf-8'))}
                data = {'chat_id': telegram_chat_id, 'caption': 'Predicted bets'}
                response = requests.post(url, data=data, files=files)
                if response.status_code != 200:
                    logging.info(f"Failed to send Telegram message: {response.text}")
            else:
                logging.info("No new rows detected. Skipping predictions.")
            previous_last_modified = current_last_modified
        except ClientError as e:
            if e.response['Error']['Code'] == 'NoSuchKey':
                logging.info("bet_data_new.csv does not exist. Sleeping.")
                time.sleep(600)
            else:
                logging.info(f"Error: {e}")
                time.sleep(600)
        except Exception as e:
            logging.info(f"Error: {e}")
            time.sleep(600)