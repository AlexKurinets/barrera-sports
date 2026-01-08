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
import io
import joblib
import torch.nn.functional as F  # Added for binary_cross_entropy_with_logits
from datetime import datetime
import logging

os.makedirs('logs', exist_ok=True)
logging.basicConfig(
    filename=f"C:/users/Administrator/barrera-sports/logs/train_lstm_fade_{datetime.now().strftime('%Y%m%d%H%M%S')}.log",
    level=logging.INFO
)

class BetDataset(Dataset):
    def __init__(self, data):
        self.data = data # now list of (seq, label, account)
    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        seq_num, seq_cat, label, account = self.data[idx]  # Updated BetDataset getitem to return seq_num, seq_cat
        return torch.tensor(seq_num, dtype=torch.float32), torch.tensor(seq_cat, dtype=torch.long), torch.tensor(label,
                                                                                                                 dtype=torch.float32), account

class LSTMModel(nn.Module):  # Updated LSTMModel to include embeddings for categorical features, added dropout=0.2 to LSTM
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

def train_batch(end_date, config, df, s3, s3_bucket):
    logging.info(f"Training LSTM model for end_date {end_date}...")
    seed = 42
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    df_batch = df[df['Date'] <= end_date].copy()
    if len(df_batch) < config['min_rows_batch']:
        print(f"Skipping end_date {end_date} due to insufficient rows: {len(df_batch)}")
        return
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
    # Preprocess
    df_batch = df_batch.sort_values(['Account', 'Date'])
    df_batch = df_batch[df_batch['W/L'].isin(['W', 'L'])]
    # Expanded numerical_cols with more raw fields
    numerical_cols = ['Decimal Odds', 'Units Risked', 'Marketed Unit', 'Unit Size', 'Wager', 'Return %', 'Net $',
                      'Account Net to Date', 'Account Average Return % to Date', 'Bet Number', 'Spread', 'Total',
                      'Odds', 'Paid $', 'Account Net to Date for Month', 'Marked Unit Qualified',
                      'Spread Bracker Qualified', 'Sport Qualified', 'League Qualified', 'Marketing Qualified',
                      'Type Qualified', 'Spread Type Qualified', 'Total Type Qualified', 'ML Type Qualified',
                      'Date Qualified', 'Count Towards Account Net', 'Last Wager in Month']
    numerical_cols = [col for col in numerical_cols if col in df_batch.columns]
    # Robust cleaning
    for col in numerical_cols:
        if col in df_batch.columns:
            df_batch[col] = pd.to_numeric(df_batch[col].astype(str).str.replace(',', '').str.replace('$', '').str.replace('%', ''),
                                          errors='coerce')
    df_batch[numerical_cols] = df_batch[numerical_cols].fillna(0)
    df_batch['Target'] = (df_batch['W/L'] == 'W').astype(float)  # Changed to binary: 1 for W, 0 for L
    logging.info(f"Win rate for {end_date}: {df_batch['Target'].mean():.4f}")  # Added print win rate to check balance
    # Add engineered features
    df_batch['Win'] = (df_batch['W/L'] == 'W').astype(float)
    df_batch['Rolling_Win_Pct_5'] = df_batch.groupby('Account')['Win'].transform(
        lambda x: x.rolling(5, min_periods=1).mean().shift(1))
    df_batch['Rolling_Win_Pct_10'] = df_batch.groupby('Account')['Win'].transform(
        lambda x: x.rolling(10, min_periods=1).mean().shift(1))
    # Additional rolling win pct
    df_batch['Rolling_Win_Pct_3'] = df_batch.groupby('Account')['Win'].transform(
        lambda x: x.rolling(3, min_periods=1).mean().shift(1))
    df_batch['Rolling_Win_Pct_15'] = df_batch.groupby('Account')['Win'].transform(
        lambda x: x.rolling(15, min_periods=1).mean().shift(1))
    df_batch['Rolling_Win_Pct_30'] = df_batch.groupby('Account')['Win'].transform(
        lambda x: x.rolling(30, min_periods=1).mean().shift(1))
    df_batch['Rolling_Win_Pct_50'] = df_batch.groupby('Account')['Win'].transform(
        lambda x: x.rolling(50, min_periods=1).mean().shift(1))
    df_batch['Rolling_Return_5'] = df_batch.groupby('Account')['Return %'].transform(
        lambda x: x.rolling(5, min_periods=1).mean().shift(1))
    df_batch['Rolling_Return_10'] = df_batch.groupby('Account')['Return %'].transform(
        lambda x: x.rolling(10, min_periods=1).mean().shift(1))
    # Additional rolling returns
    df_batch['Rolling_Return_3'] = df_batch.groupby('Account')['Return %'].transform(
        lambda x: x.rolling(3, min_periods=1).mean().shift(1))
    df_batch['Rolling_Return_15'] = df_batch.groupby('Account')['Return %'].transform(
        lambda x: x.rolling(15, min_periods=1).mean().shift(1))
    df_batch['Rolling_Return_30'] = df_batch.groupby('Account')['Return %'].transform(
        lambda x: x.rolling(30, min_periods=1).mean().shift(1))
    df_batch['Rolling_Return_50'] = df_batch.groupby('Account')['Return %'].transform(
        lambda x: x.rolling(50, min_periods=1).mean().shift(1))
    # Rolling net sums
    df_batch['Rolling_Net_5'] = df_batch.groupby('Account')['Net $'].transform(
        lambda x: x.rolling(5, min_periods=1).sum().shift(1))
    df_batch['Rolling_Net_10'] = df_batch.groupby('Account')['Net $'].transform(
        lambda x: x.rolling(10, min_periods=1).sum().shift(1))
    df_batch['Rolling_Net_20'] = df_batch.groupby('Account')['Net $'].transform(
        lambda x: x.rolling(20, min_periods=1).sum().shift(1))
    df_batch['Rolling_Net_30'] = df_batch.groupby('Account')['Net $'].transform(
        lambda x: x.rolling(30, min_periods=1).sum().shift(1))
    # Rolling odds means
    df_batch['Rolling_Odds_5'] = df_batch.groupby('Account')['Decimal Odds'].transform(
        lambda x: x.rolling(5, min_periods=1).mean().shift(1))
    df_batch['Rolling_Odds_10'] = df_batch.groupby('Account')['Decimal Odds'].transform(
        lambda x: x.rolling(10, min_periods=1).mean().shift(1))
    # Rolling return std
    df_batch['Rolling_Return_Std_5'] = df_batch.groupby('Account')['Return %'].transform(
        lambda x: x.rolling(5, min_periods=1).std().shift(1))
    df_batch['Rolling_Return_Std_10'] = df_batch.groupby('Account')['Return %'].transform(
        lambda x: x.rolling(10, min_periods=1).std().shift(1))
    # EWM
    df_batch['EWM_Win_Pct'] = df_batch.groupby('Account')['Win'].apply(lambda x: x.ewm(span=10, adjust=False).mean()).groupby(
        level=0).shift(1).droplevel(0)
    df_batch['EWM_Return'] = df_batch.groupby('Account')['Return %'].apply(lambda x: x.ewm(span=10, adjust=False).mean()).groupby(level=0).shift(1).droplevel(0)
    # Lags
    df_batch['Lag1_Return'] = df_batch.groupby('Account')['Return %'].shift(1)
    df_batch['Lag1_Net'] = df_batch.groupby('Account')['Net $'].shift(1)
    # Streak
    df_batch['streak_group'] = (df_batch.groupby('Account')['Win'].diff() != 0).cumsum()
    df_batch['Streak'] = df_batch.groupby(['Account', 'streak_group']).cumcount() + 1
    df_batch['Streak'] = np.where(df_batch['Win'] == 0, -df_batch['Streak'], df_batch['Streak'])
    df_batch = df_batch.drop('streak_group', axis=1)
    # Days since last
    df_batch['Days_Since_Last'] = df_batch.groupby('Account')['Date'].diff().dt.days
    # Rolling days between
    df_batch['Rolling_Days_Between_5'] = df_batch.groupby('Account')['Days_Since_Last'].transform(
        lambda x: x.rolling(5, min_periods=1).mean().shift(1))
    # # Time features
    df_batch['Day_of_Week'] = df_batch['Date'].dt.dayofweek
    df_batch['Is_Weekend'] = (df_batch['Day_of_Week'] >= 5).astype(int)
    df_batch['Month'] = df_batch['Date'].dt.month
    # Logs
    df_batch['Bet_Number_Log'] = np.log1p(df_batch['Bet Number'])
    df_batch['Wager_Log'] = np.log1p(df_batch['Wager'])
    df_batch['Days'] = (df_batch['Date'] - df_batch.groupby('Account')['Date'].transform('min')).dt.days
    # Fill NaNs for new features
    new_cols = ['Rolling_Win_Pct_5', 'Rolling_Win_Pct_10', 'Rolling_Win_Pct_3', 'Rolling_Win_Pct_15',
                'Rolling_Win_Pct_30', 'Rolling_Win_Pct_50',
                'Rolling_Return_5', 'Rolling_Return_10', 'Rolling_Return_3', 'Rolling_Return_15', 'Rolling_Return_30',
                'Rolling_Return_50',
                'Rolling_Net_5', 'Rolling_Net_10', 'Rolling_Net_20', 'Rolling_Net_30',
                'Rolling_Odds_5', 'Rolling_Odds_10', 'Rolling_Return_Std_5', 'Rolling_Return_Std_10',
                'EWM_Win_Pct', 'EWM_Return', 'Lag1_Return', 'Lag1_Net', 'Streak', 'Days_Since_Last',
                'Rolling_Days_Between_5',
                'Day_of_Week', 'Is_Weekend', 'Month',
                'Bet_Number_Log', 'Wager_Log'
                ]
    new_cols.append('Days')
    df_batch[new_cols] = df_batch[new_cols].fillna(0)
    end_date_str = end_date.strftime('%Y%m%d')
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
    # Add to numerical_cols
    numerical_cols += new_cols
    # Expanded categorical
    categorical_cols = ['Type', 'Sport', 'League', 'Marketing', 'Spread Type', 'Total Type', 'ML Type',
                        'Odds Bracket', 'Wager Bracket', 'Team', 'Division']
    categorical_cols = [col for col in categorical_cols if col in df_batch.columns]
    encoders = {col: LabelEncoder().fit(df_batch[col].astype(str)) for col in categorical_cols}
    for col in categorical_cols:
        df_batch[col] = encoders[col].transform(df_batch[col].astype(str))
    scaler = StandardScaler().fit(df_batch[numerical_cols])
    df_batch[numerical_cols] = scaler.transform(df_batch[numerical_cols])
    preprocessors = {'scaler': scaler, 'encoders': encoders}
    joblib_filename = (f'c:/users/administrator/barrera-sports/output/models/'
                       f'preprocessors_{hyperparams_str}_end_{end_date_str}.joblib')
    os.makedirs(os.path.dirname(joblib_filename), exist_ok=True)
    joblib.dump(preprocessors, joblib_filename)
    s3_key_joblib = 'models/' + os.path.basename(joblib_filename)
    with io.BytesIO() as buffer:
        joblib.dump(preprocessors, buffer)
        buffer.seek(0)
        s3.upload_fileobj(buffer, s3_bucket, s3_key_joblib)

    # Build sequences per account
    num_features = numerical_cols  # In sequence building, separated seq_num and seq_cat
    cat_features = categorical_cols
    data = []
    groups = df_batch.groupby('Account')
    for name, group in groups:
        for i in range(0, len(group) - sequence_length, steps_ahead):
            seq_num = group.iloc[i:i + sequence_length][num_features].values
            seq_cat = group.iloc[i:i + sequence_length][cat_features].values
            label = group.iloc[i + sequence_length]['Target']
            data.append((seq_num, seq_cat, label, name))
    # Dataset
    train_data, val_data = train_test_split(data, test_size=0.2, random_state=42)
    train_dataset = BetDataset(train_data)
    val_dataset = BetDataset(val_data)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    # Model
    vocab_sizes = {col: len(encoders[col].classes_) + 1 for col in categorical_cols}
    embed_dim = 16
    input_size_num = len(num_features)
    model = LSTMModel(input_size_num, categorical_cols, vocab_sizes, embed_dim, hidden_size, num_layers)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=l2_reg)
    log_dir = (f"c:/users/administrator/barrera-sports/output/tensorboard/runs/"
               f"{hyperparams_str}/{hyperparams_str}_end_{end_date_str}")
    os.makedirs(log_dir, exist_ok=True)
    writer = SummaryWriter(log_dir=log_dir)
    min_val_loss = float('inf')
    counter = 0
    best_model_state = None
    for epoch in range(max_epochs):
        model.train()
        train_loss = 0.0
        for seqs_num, seqs_cat, labels, _ in train_loader:  # In training and val loops, passed seqs_num, seqs_cat to model
            seqs_num = seqs_num.to(device)
            seqs_cat = seqs_cat.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            outputs = model(seqs_num, seqs_cat).view(-1)
            loss = criterion(outputs, labels)
            if l1_reg > 0:
                l1_penalty = sum(p.abs().sum() for p in model.parameters())
                loss += l1_reg * l1_penalty
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * seqs_num.size(0)
        train_loss /= len(train_loader.dataset)
        writer.add_scalar('Loss/train', train_loss, epoch)
        model.eval()
        account_losses = {}
        account_counts = {}
        total_val_loss = 0.0
        with torch.no_grad():
            for seqs_num, seqs_cat, labels, accounts in val_loader:
                seqs_num = seqs_num.to(device)
                seqs_cat = seqs_cat.to(device)
                labels = labels.to(device)
                outputs = model(seqs_num, seqs_cat).view(-1)
                for out, lab, acc in zip(outputs, labels, accounts):
                    loss_item = F.binary_cross_entropy_with_logits(out, lab,
                                                                   reduction='sum').item()  # Changed from MSE to BCE
                    total_val_loss += loss_item
                    if acc not in account_losses:
                        account_losses[acc] = 0.0
                        account_counts[acc] = 0
                    account_losses[acc] += loss_item
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
        logging.info(f'Epoch {epoch + 1}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f} for end_date {end_date}')
        if val_loss < min_val_loss:
            min_val_loss = val_loss
            counter = 0
            best_model_state = model.state_dict().copy()
        else:
            counter += 1
            if counter >= patience:
                print(f'Early stopping at epoch {epoch + 1} for end_date {end_date}')
                break
    writer.close()
    # Save best model
    filename = (f'c:/users/administrator/barrera-sports/'
                f'output/models/'
                f'fade_model_seq_{sequence_length}_'
                f'bs_{batch_size}_'
                f'hs_{hidden_size}_'
                f'nl_{num_layers}_'
                f'lr_{lr}_'
                f'ep_{max_epochs}_'
                f'pat_{patience}_'
                f'end_{end_date_str}.pth')
    torch.save(best_model_state, filename)
    s3_key_model = 'models/' + os.path.basename(filename)
    with io.BytesIO() as buffer:
        torch.save(best_model_state, buffer)
        buffer.seek(0)
        s3.upload_fileobj(buffer, s3_bucket, s3_key_model)

if __name__ == "__main__":
    config = {
        'sequence_length': 40,
        'batch_size': 32,
        'hidden_size': 32,
        'num_layers': 2,
        'l1_regularization': 0,
        'l2_regularization': 0,
        'steps_ahead': 10,
        'learning_rate': 0.001,
        'patience': 10,
        'max_epochs': 40,
        'min_rows_batch': 300
    }
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
    s3_key = 'bet_data/bet_data_new.csv'
    obj = s3.get_object(Bucket=s3_bucket, Key=s3_key)
    df = pd.read_csv(obj['Body'], encoding='latin1')
    df["Date"] = pd.to_datetime(df["Date"], format = "%Y-%m-%d")
    cum_rows = df.groupby('Date').size().sort_index().cumsum()
    unique_dates = cum_rows.index.tolist()
    min_rows_batch = config['min_rows_batch']
    end_dates = [d for d in unique_dates if cum_rows[d] >= min_rows_batch]
    # Sequential processing
    for end_date in end_dates[-1:]: # Use only the last end_date for single model
        train_batch(end_date, config, df, s3, s3_bucket)