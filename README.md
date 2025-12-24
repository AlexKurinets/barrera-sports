# Betting Data Ingest and LSTM Training Pipeline

## Overview
This repository contains two Python scripts for managing and modeling sports betting data: `data_ingest_to_aws.py` for ingesting data from Google Sheets to AWS S3, and `train_lstm_fade.py` for training an LSTM neural network to predict betting outcomes, specifically a "fade" signal (targeting potential losses). The pipeline supports data appending, preprocessing, feature engineering, and model training with TensorBoard logging. Dependencies include pandas, numpy, torch, sklearn, boto3, and requests. Requires AWS credentials stored in Secrets Manager.

## Data Ingest Script: data_ingest_to_aws.py
This script automates the ingestion of betting data from a public Google Sheet to an S3 bucket, ensuring only new records are appended.

- **Process**:
  1. Retrieves AWS credentials from Secrets Manager (secret: 'secrets_key').
  2. Downloads CSV from Google Sheet (ID: '1wM00QhayythUnOizO7hIzv4hDHQutmTS', GID: '1467560424').
  3. Converts 'Date' to datetime.
  4. Checks for existing CSV in S3 (bucket from secrets, key: 'bet_data/bet_data.csv').
  5. Appends rows with dates after the max in existing data.
  6. Uploads updated CSV if new records exist.

- **Usage**: Run as standalone: `python data_ingest_to_aws.py`. Outputs number of new records loaded or "No new records."
- **Requirements**: AWS account with S3 and Secrets Manager access. Google Sheet must be publicly exportable.

This ensures a growing historical dataset for modeling, with incremental updates to avoid redundancy.

## Training Script: train_lstm_fade.py
This script fetches betting data from S3, preprocesses it, engineers features, and trains an LSTM model to predict a 'Target' variable: (Decimal Odds - 1) for losses ('L'), -1 for wins ('W'). The model uses sequences of past bets per account to forecast fade opportunities. Configurable hyperparameters include sequence_length=10, batch_size=32, hidden_size=64, num_layers=2, l1/l2 regularization=0.001 each, steps_ahead=5, learning_rate=0.001, patience=10, max_epochs=40, min_rows=300.

- **Data Fetch and Preprocessing**:
  1. Loads CSV from S3 using secrets.
  2. Skips if fewer than min_rows.
  3. Filters to 'W'/'L' outcomes, sorts by Account/Date.
  4. Cleans numerical columns (e.g., remove commas/$/%), fills NaNs with 0.
  5. Computes Target.

- **Feature Engineering** (Predictor Variables):
  Categorical (label-encoded): Type, Sport, League, Marketing, Spread Type, Total Type, ML Type, Odds Bracket, Wager Bracket, Team, Division.
  Numerical (standard-scaled): Decimal Odds, Units Risked, Marketed Unit, Unit Size, Wager, Return %, Net $, Account Net to Date, Account Average Return % to Date, Bet Number, Spread, Total, Odds, Paid $, Account Net to Date for Month, various Qualified flags (Marked Unit, Spread Bracket, Sport, etc.), Count Towards Account Net, Last Wager in Month.
  Engineered:
  - Rolling win percentages (windows: 3,5,10,15,30,50).
  - Rolling returns means (windows: 3,5,10,15,30,50).
  - Rolling net sums (windows: 5,10,20,30).
  - Rolling odds means (windows: 5,10).
  - Rolling return stds (windows: 5,10).
  - Exponential weighted moving averages: EWM_Win_Pct, EWM_Return (span=10).
  - Lags: Lag1_Return, Lag1_Net.
  - Streak: Current win/loss streak.
  - Time-based: Days_Since_Last, Rolling_Days_Between_5, Day_of_Week, Is_Weekend, Month.
  - Logs: Bet_Number_Log, Wager_Log.
  Plus 'Days' (days since first bet per account) in sequences.

- **Model Training**:
  1. Builds sequences per account (length=sequence_length, step=steps_ahead).
  2. Splits 80/20 train/val.
  3. LSTM model: input_size = features + 1 (Days), outputs single value.
  4. Trains with MSE loss, Adam optimizer, L1/L2 reg.
  5. Logs to TensorBoard (per-account val losses).
  6. Early stopping on val loss.
  7. Saves best model to 'output/models/' with hyperparam-named .pth.

- **Usage**: Run `python train_lstm_fade.py`. View logs in TensorBoard: `tensorboard --logdir=output/tensorboard/runs`.
- **Notes**: Seed=42 for reproducibility. CUDA if available.

This pipeline enables predictive analytics on betting performance, focusing on account-specific patterns for fade strategies.