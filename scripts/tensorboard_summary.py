import json
import boto3
import pandas as pd
import numpy as np
import os
import glob
from datetime import datetime
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

# Get secrets
secrets_client = boto3.client('secretsmanager', region_name='us-east-2')
secret_name = 'secrets_key'
response = secrets_client.get_secret_value(SecretId=secret_name)
secrets = json.loads(response['SecretString'])
aws_access_key = secrets['aws_access_key_id']
aws_secret_key = secrets['aws_secret_access_key']
s3_bucket = secrets['bucket_name']

# Fetch data from S3
s3 = boto3.client('s3', aws_access_key_id=aws_access_key, aws_secret_access_key=aws_secret_key)
s3_key = 'bet_data/bet_data.csv'
obj = s3.get_object(Bucket=s3_bucket, Key=s3_key)
df = pd.read_csv(obj['Body'], encoding='latin1')
df['Date'] = pd.to_datetime(df['Date'])
start_date = df['Date'].min().strftime('%Y%m%d')

root = 'output/tensorboard/runs'
results = []

for hyper_dir in os.listdir(root):
    hyper_path = os.path.join(root, hyper_dir)
    if os.path.isdir(hyper_path):
        # Parse hyperparams
        parts = hyper_dir.split('_')
        hyperparams = {
            'sequence_length': int(parts[1]),
            'batch_size': int(parts[3]),
            'hidden_size': int(parts[5]),
            'num_layers': int(parts[7]),
            'l1_regularization': float(parts[9]),
            'l2_regularization': float(parts[11]),
            'steps_ahead': int(parts[13]),
            'learning_rate': float(parts[15]),
            'patience': int(parts[17]),
            'max_epochs': int(parts[19])
        }
        for end_subdir in os.listdir(hyper_path):
            end_path = os.path.join(hyper_path, end_subdir)
            if os.path.isdir(end_path):
                end_parts = end_subdir.split('_end_')
                end_date = end_parts[-1] if len(end_parts) > 1 else ''
                event_files = glob.glob(os.path.join(end_path, 'events.out.tfevents.*'))
                if event_files:
                    event_file = event_files[0]
                    ea = EventAccumulator(event_file)
                    ea.Reload()
                    tags = ea.Tags()['scalars']
                    account_tags = [t for t in tags if t.startswith('Loss/val_account_')]
                    for tag in account_tags:
                        account = tag.split('Loss/val_account_')[1]
                        scalars = ea.Scalars(tag)
                        values = [s.value for s in scalars]
                        if values:
                            avg = np.mean(values)
                            stdev = np.std(values)
                            min_val = np.min(values)
                            max_val = np.max(values)
                            result = hyperparams.copy()
                            result.update({
                                'start_date': start_date,
                                'end_date': end_date,
                                'account': account,
                                'avg_val_loss': avg,
                                'stdev_val_loss': stdev,
                                'min_val_loss': min_val,
                                'max_val_loss': max_val
                            })
                            results.append(result)

result_df = pd.DataFrame(results)
group_cols = ['account', 'sequence_length', 'batch_size', 'hidden_size', 'num_layers',
              'l1_regularization', 'l2_regularization', 'steps_ahead', 'learning_rate',
              'patience', 'max_epochs']
agg_dict = {
    'avg_val_loss': 'mean',
    'stdev_val_loss': 'mean',
    'min_val_loss': 'mean',
    'max_val_loss': 'mean',
    'start_date': 'first',
    'end_date': 'max',
}
results_processed_df = result_df.groupby(group_cols).agg(agg_dict).reset_index()
results_processed_df.sort_values([ 'avg_val_loss'], ascending=True, inplace=True)
results_processed_df.to_csv(f"output/tensorboard/summary/results_processed_{datetime.now().strftime('%Y%m%d')}.csv", index=False)
hyper_cols = ['sequence_length', 'batch_size', 'hidden_size', 'num_layers', 'l1_regularization',
              'l2_regularization', 'steps_ahead', 'learning_rate', 'patience', 'max_epochs']
results_processed_df['rank'] = results_processed_df.groupby('account')['avg_val_loss'].rank(method='min', ascending=True)
rank_df = results_processed_df.groupby(hyper_cols)['rank'].mean().reset_index()
rank_df.rename(columns={'rank': 'avg_rank'}, inplace=True)
min_start = results_processed_df['start_date'].min()
max_end = results_processed_df['end_date'].max()
rank_df['start_date'] = min_start
rank_df['end_date'] = max_end
rank_df.sort_values('avg_rank', ascending=True, inplace=True)
rank_df.to_csv(f"output/tensorboard/summary/rank_results_processed_{datetime.now().strftime('%Y%m%d')}.csv", index=False)
print(rank_df)
print(results_processed_df)