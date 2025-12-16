import json
import boto3
import pandas as pd
import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LinearRegression

class BetDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        seq, label = self.data[idx]
        return torch.tensor(seq, dtype=torch.float32), torch.tensor(label, dtype=torch.float32)


class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        _, (h, _) = self.lstm(x)
        out = self.fc(h[-1])
        return out

if __name__ == "__main__":
    print("Training LSTM model...")
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
    s3_key = 'bet_data.csv'
    obj = s3.get_object(Bucket=s3_bucket, Key=s3_key)
    df = pd.read_csv(obj['Body'])
    df['Date'] = pd.to_datetime(df['Date'])

    # Preprocess
    df = df.sort_values(['Account', 'Date'])
    df = df[df['W/L'].isin(['W', 'L'])]
    # Clean numerical columns before target calculation
    numerical_cols = ['Decimal Odds', 'Units Risked', 'Marketed Unit', 'Unit Size', 'Wager', 'Return %', 'Net $',
                      'Account Net to Date', 'Account Average Return % to Date', 'Bet Number']
    for col in numerical_cols:
        if df[col].dtype == 'object':
            df[col] = df[col].str.replace(',', '').str.replace('$', '').str.replace('%', '').astype(float)
    df['Target'] = np.where(df['W/L'] == 'L', df['Decimal Odds'] - 1, -1)
    # Add engineered features
    df['Win'] = (df['W/L'] == 'W').astype(float)
    df['Rolling_Win_Pct_5'] = df.groupby('Account')['Win'].transform(
        lambda x: x.rolling(5, min_periods=1).mean().shift(1))
    df['Rolling_Win_Pct_10'] = df.groupby('Account')['Win'].transform(
        lambda x: x.rolling(10, min_periods=1).mean().shift(1))
    df['Rolling_Return_5'] = df.groupby('Account')['Return %'].transform(
        lambda x: x.rolling(5, min_periods=1).mean().shift(1))
    df['Rolling_Return_10'] = df.groupby('Account')['Return %'].transform(
        lambda x: x.rolling(10, min_periods=1).mean().shift(1))
    # Fill NaN in rolling features
    rolling_cols = ['Rolling_Win_Pct_5', 'Rolling_Win_Pct_10', 'Rolling_Return_5', 'Rolling_Return_10']
    df[rolling_cols] = df[rolling_cols].fillna(0)


    # Trendline deviation
    def calc_trend_dev(group):
        if len(group) < 2:
            group['Trend_Dev'] = 0
            return group
        x = group['Bet Number'].values.reshape(-1, 1)
        y = group['Account Net to Date'].values
        reg = LinearRegression().fit(x, y)
        pred = reg.predict(x)
        group['Trend_Dev'] = y - pred
        return group


    df = df.groupby('Account').apply(calc_trend_dev).reset_index(drop=True)
    # Add new features to numerical_cols
    numerical_cols += rolling_cols + ['Trend_Dev']
    categorical_cols = ['Type', 'Sport', 'League', 'Marketing', 'Spread Type', 'Total Type', 'ML Type', 'Odds Bracket', 'Wager Bracket']

    encoders = {col: LabelEncoder().fit(df[col].astype(str)) for col in categorical_cols}
    for col in categorical_cols:
        df[col] = encoders[col].transform(df[col].astype(str))

    scaler = StandardScaler().fit(df[numerical_cols])
    df[numerical_cols] = scaler.transform(df[numerical_cols])

    # Build sequences per account
    sequence_length = 10
    features = categorical_cols + numerical_cols
    data = []
    groups = df.groupby('Account')
    for name, group in groups:
        group['Days'] = (group['Date'] - group['Date'].min()).dt.days
        for i in range(len(group) - sequence_length):
            seq = group.iloc[i:i+sequence_length][features + ['Days']].values
            label = group.iloc[i+sequence_length]['Target']
            data.append((seq, label))

    # Dataset
    dataset = BetDataset(data)
    loader = DataLoader(dataset, batch_size=32, shuffle=True)

    # Model
    input_size = len(features) + 1  # + Days
    model = LSTMModel(input_size, 64, 2)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # Train
    for epoch in range(10):
        model.train()
        for seqs, labels in loader:
            seqs = seqs.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            outputs = model(seqs).squeeze()
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
        print(f'Epoch {epoch+1}, Loss: {loss.item():.4f}')

    # Save model
    torch.save(model.state_dict(), 'output/models/fade_model.pth')