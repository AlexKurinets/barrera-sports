import json
import boto3
import pandas as pd
import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler, LabelEncoder
from torch.utils.tensorboard import SummaryWriter
from sklearn.model_selection import train_test_split
import random
import os

class BetDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        seq, label, account = self.data[idx]
        return torch.tensor(seq, dtype=torch.float32), torch.tensor(label, dtype=torch.float32), account

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
    config = {
        'sequence_length': 10,
        'batch_size': 32,
        'hidden_size': 64,
        'num_layers': 2,
        'l1_regularization': 0.001,
        'l2_regularization': 0.001,
        'steps_ahead': 1,
        'learning_rate': 0.001,
        'patience': 10,
        'max_epochs': 40,
        'min_rows': 300
    }
    seed = 42
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    sequence_length = config['sequence_length']
    batch_size = config['batch_size']
    hidden_size = config['hidden_size']
    num_layers = config['num_layers']
    l1_reg = config['l1_regularization']
    l2_reg = config['l2_regularization']
    steps_ahead = config['steps_ahead']
    lr = config['learning_rate']
    patience = config['patience']
    max_epochs = config['max_epochs']
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
    if len(df) < config['min_rows']:
        print(f"Skipping due to insufficient rows: {len(df)}")
        import sys

        sys.exit(0)
    # Preprocess
    df = df.sort_values(['Account', 'Date'])
    df = df[df['W/L'].isin(['W', 'L'])]
    # Expanded numerical_cols with more raw fields
    numerical_cols = ['Decimal Odds', 'Units Risked', 'Marketed Unit', 'Unit Size', 'Wager', 'Return %', 'Net $',
                      'Account Net to Date', 'Account Average Return % to Date', 'Bet Number', 'Spread', 'Total',
                      'Odds', 'Paid $', 'Account Net to Date for Month', 'Marked Unit Qualified',
                      'Spread Bracker Qualified', 'Sport Qualified', 'League Qualified', 'Marketing Qualified',
                      'Type Qualified', 'Spread Type Qualified', 'Total Type Qualified', 'ML Type Qualified',
                      'Date Qualified', 'Count Towards Account Net', 'Last Wager in Month']
    numerical_cols = [col for col in numerical_cols if col in df.columns]
    # Robust cleaning
    for col in numerical_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col].astype(str).str.replace(',', '').str.replace('$', '').str.replace('%', ''),
                                    errors='coerce')
    df[numerical_cols] = df[numerical_cols].fillna(0)
    df['Target'] = np.where(df['W/L'] == 'L', df['Decimal Odds'] - 1, -1)
    # Add engineered features
    df['Win'] = (df['W/L'] == 'W').astype(float)
    df['Rolling_Win_Pct_5'] = df.groupby('Account')['Win'].transform(
        lambda x: x.rolling(5, min_periods=1).mean().shift(1))
    df['Rolling_Win_Pct_10'] = df.groupby('Account')['Win'].transform(
        lambda x: x.rolling(10, min_periods=1).mean().shift(1))
    # Additional rolling win pct
    df['Rolling_Win_Pct_3'] = df.groupby('Account')['Win'].transform(
        lambda x: x.rolling(3, min_periods=1).mean().shift(1))
    df['Rolling_Win_Pct_15'] = df.groupby('Account')['Win'].transform(
        lambda x: x.rolling(15, min_periods=1).mean().shift(1))
    df['Rolling_Win_Pct_30'] = df.groupby('Account')['Win'].transform(
        lambda x: x.rolling(30, min_periods=1).mean().shift(1))
    df['Rolling_Win_Pct_50'] = df.groupby('Account')['Win'].transform(
        lambda x: x.rolling(50, min_periods=1).mean().shift(1))
    df['Rolling_Return_5'] = df.groupby('Account')['Return %'].transform(
        lambda x: x.rolling(5, min_periods=1).mean().shift(1))
    df['Rolling_Return_10'] = df.groupby('Account')['Return %'].transform(
        lambda x: x.rolling(10, min_periods=1).mean().shift(1))

    # Additional rolling returns
    df['Rolling_Return_3'] = df.groupby('Account')['Return %'].transform(
        lambda x: x.rolling(3, min_periods=1).mean().shift(1))
    df['Rolling_Return_15'] = df.groupby('Account')['Return %'].transform(
        lambda x: x.rolling(15, min_periods=1).mean().shift(1))
    df['Rolling_Return_30'] = df.groupby('Account')['Return %'].transform(
        lambda x: x.rolling(30, min_periods=1).mean().shift(1))
    df['Rolling_Return_50'] = df.groupby('Account')['Return %'].transform(
        lambda x: x.rolling(50, min_periods=1).mean().shift(1))

    # Rolling net sums
    df['Rolling_Net_5'] = df.groupby('Account')['Net $'].transform(
        lambda x: x.rolling(5, min_periods=1).sum().shift(1))
    df['Rolling_Net_10'] = df.groupby('Account')['Net $'].transform(
        lambda x: x.rolling(10, min_periods=1).sum().shift(1))
    df['Rolling_Net_20'] = df.groupby('Account')['Net $'].transform(
        lambda x: x.rolling(20, min_periods=1).sum().shift(1))
    df['Rolling_Net_30'] = df.groupby('Account')['Net $'].transform(
        lambda x: x.rolling(30, min_periods=1).sum().shift(1))

    # Rolling odds means
    df['Rolling_Odds_5'] = df.groupby('Account')['Decimal Odds'].transform(
        lambda x: x.rolling(5, min_periods=1).mean().shift(1))
    df['Rolling_Odds_10'] = df.groupby('Account')['Decimal Odds'].transform(
        lambda x: x.rolling(10, min_periods=1).mean().shift(1))

    # Rolling return std
    df['Rolling_Return_Std_5'] = df.groupby('Account')['Return %'].transform(
        lambda x: x.rolling(5, min_periods=1).std().shift(1))
    df['Rolling_Return_Std_10'] = df.groupby('Account')['Return %'].transform(
        lambda x: x.rolling(10, min_periods=1).std().shift(1))

    # EWM
    df['EWM_Win_Pct'] = df.groupby('Account')['Win'].apply(lambda x: x.ewm(span=10, adjust=False).mean()).groupby(
        level=0).shift(1).droplevel(0)
    df['EWM_Return'] = df.groupby('Account')['Return %'].apply(lambda x: x.ewm(span=10, adjust=False).mean()).groupby(level=0).shift(1).droplevel(0)

    # Lags
    df['Lag1_Return'] = df.groupby('Account')['Return %'].shift(1)
    df['Lag1_Net'] = df.groupby('Account')['Net $'].shift(1)

    # Streak
    df['streak_group'] = (df.groupby('Account')['Win'].diff() != 0).cumsum()
    df['Streak'] = df.groupby(['Account', 'streak_group']).cumcount() + 1
    df['Streak'] = np.where(df['Win'] == 0, -df['Streak'], df['Streak'])
    df = df.drop('streak_group', axis=1)

    # Days since last
    df['Days_Since_Last'] = df.groupby('Account')['Date'].diff().dt.days

    # Rolling days between
    df['Rolling_Days_Between_5'] = df.groupby('Account')['Days_Since_Last'].transform(
        lambda x: x.rolling(5, min_periods=1).mean().shift(1))

    # Time features
    df['Day_of_Week'] = df['Date'].dt.dayofweek
    df['Is_Weekend'] = (df['Day_of_Week'] >= 5).astype(int)
    df['Month'] = df['Date'].dt.month

    # Logs
    df['Bet_Number_Log'] = np.log1p(df['Bet Number'])
    df['Wager_Log'] = np.log1p(df['Wager'])
    # Fill NaNs for new features
    new_cols = ['Rolling_Win_Pct_3', 'Rolling_Win_Pct_15', 'Rolling_Win_Pct_30', 'Rolling_Win_Pct_50',
                'Rolling_Return_3', 'Rolling_Return_15', 'Rolling_Return_30', 'Rolling_Return_50',
                'Rolling_Net_5', 'Rolling_Net_10', 'Rolling_Net_20', 'Rolling_Net_30',
                'Rolling_Odds_5', 'Rolling_Odds_10', 'Rolling_Return_Std_5', 'Rolling_Return_Std_10',
                'EWM_Win_Pct', 'EWM_Return', 'Lag1_Return', 'Lag1_Net', 'Streak', 'Days_Since_Last',
                'Rolling_Days_Between_5', 'Day_of_Week', 'Is_Weekend', 'Month', 'Bet_Number_Log', 'Wager_Log']
    df[new_cols] = df[new_cols].fillna(0)
    # Add to numerical_cols
    numerical_cols += new_cols
    # Expanded categorical
    categorical_cols = ['Type', 'Sport', 'League', 'Marketing', 'Spread Type', 'Total Type', 'ML Type',
                        'Odds Bracket', 'Wager Bracket', 'Team', 'Division']
    categorical_cols = [col for col in categorical_cols if col in df.columns]
    encoders = {col: LabelEncoder().fit(df[col].astype(str)) for col in categorical_cols}
    for col in categorical_cols:
        df[col] = encoders[col].transform(df[col].astype(str))

    scaler = StandardScaler().fit(df[numerical_cols])
    df[numerical_cols] = scaler.transform(df[numerical_cols])

    # Build sequences per account
    features = categorical_cols + numerical_cols
    data = []
    groups = df.groupby('Account')
    for name, group in groups:
        group['Days'] = (group['Date'] - group['Date'].min()).dt.days
        for i in range(0, len(group) - sequence_length, steps_ahead):
            seq = group.iloc[i:i+sequence_length][features + ['Days']].values
            label = group.iloc[i+sequence_length]['Target']
            data.append((seq, label, name))

    # Dataset
    train_data, val_data = train_test_split(data, test_size=0.2, random_state=42)
    train_dataset = BetDataset(train_data)
    val_dataset = BetDataset(val_data)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

    # Model
    input_size = len(features) + 1  # + Days
    model = LSTMModel(input_size, 64, 2)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=l2_reg)

    hyperparams_str = (f"seq_{sequence_length}"
                       f"_bs_{batch_size}"
                       f"_hs_{hidden_size}"
                       f"_nl_{num_layers}"
                       f"_l1_{l1_reg}"
                       f"_l2_{l2_reg}"
                       f"_sa_{steps_ahead}"
                       f"_lr_{lr}"
                       f"_pat_{patience}"
                       f"_ep_{max_epochs}")
    log_dir = f"output/tensorboard/runs/{hyperparams_str}/{hyperparams_str}"
    os.makedirs(log_dir, exist_ok=True)
    writer = SummaryWriter(log_dir=log_dir)
    min_val_loss = float('inf')
    counter = 0
    best_model_state = None
    for epoch in range(max_epochs):
        model.train()
        train_loss = 0.0
        for seqs, labels, _ in train_loader:
            seqs = seqs.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            outputs = model(seqs).squeeze()
            loss = criterion(outputs, labels)
            if l1_reg > 0:
                l1_penalty = sum(p.abs().sum() for p in model.parameters())
                loss += l1_reg * l1_penalty
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * seqs.size(0)
        train_loss /= len(train_loader.dataset)
        writer.add_scalar('Loss/train', train_loss, epoch)

        model.eval()
        account_losses = {}
        account_counts = {}
        total_val_loss = 0.0
        with torch.no_grad():
            for seqs, labels, accounts in val_loader:
                seqs = seqs.to(device)
                labels = labels.to(device)
                outputs = model(seqs).squeeze()
                for out, lab, acc in zip(outputs, labels, accounts):
                    sq_err = ((out - lab) ** 2).item()
                    total_val_loss += sq_err
                    if acc not in account_losses:
                        account_losses[acc] = 0.0
                        account_counts[acc] = 0
                    account_losses[acc] += sq_err
                    account_counts[acc] += 1
        if sum(account_counts.values()) > 0:
            val_loss = total_val_loss / sum(account_counts.values())
        else:
            val_loss = 0.0
        writer.add_scalar('Loss/val', val_loss, epoch)
        for acc in account_losses:
            if account_counts[acc] > 0:
                acc_val_loss = account_losses[acc] / account_counts[acc]
                writer.add_scalar(f'Loss/val_account_{acc}', acc_val_loss, epoch)
        print(f'Epoch {epoch + 1}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')

        if val_loss < min_val_loss:
            min_val_loss = val_loss
            counter = 0
            best_model_state = model.state_dict().copy()
        else:
            counter += 1
            if counter >= patience:
                print(f'Early stopping at epoch {epoch + 1}')
                break
    writer.close()
    # Save best model
    filename = (f'output/models/fade_model_seq_{sequence_length}_'
                f'bs_{batch_size}_'
                f'hs_{hidden_size}_'
                f'nl_{num_layers}_'
                f'lr_{lr}_'
                f'ep_{max_epochs}_'
                f'pat_{patience}.pth')
    torch.save(best_model_state, filename)