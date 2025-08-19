import logging
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

logger = logging.getLogger(__name__)
RANDOM_STATE = 42
TARGET_COL = 'target'

scaler = None
numerical_columns = []

def add_time_features(df):

    logger.info("Adding time features")
    
    if 'transaction_time' in df.columns:
        df['transaction_time'] = pd.to_datetime(df['transaction_time'])
        df['year'] = df['transaction_time'].dt.year
        df['month'] = df['transaction_time'].dt.month
        df['hour'] = df['transaction_time'].dt.hour
        df['day_of_week'] = df['transaction_time'].dt.weekday
        df['is_night'] = df['hour'].apply(lambda x: 1 if (x >= 22 or x <= 3) else 0)
        
    return df

def preprocess_data(input_df):

    global scaler, numerical_columns
    logger.info("Starting preprocessing")
    input_df = add_time_features(input_df)
    
    cols_drop = [
         'lat', 'lon', 'post_code', 'year', 'transaction_time', 'name_1', 'name_2'
    ]

    input_df = input_df.drop(columns=cols_drop, errors='ignore')
    logger.info(f"After dropping columns: {input_df.shape}")
    cats = ['street', 'one_city', 'us_state', 'jobs', 'merch', 'cat_id', 'gender']

    if TARGET_COL in numerical_columns:
            numerical_columns.remove(TARGET_COL)
    
    logger.info("Preprocessing complete")
    return input_df, cats

def load_train_data(path='./train_data/train.csv'):

    global scaler
    logger.info('Loading training data...')
    train_df = pd.read_csv(path)
    logger.info('Raw train data imported. Shape: %s', train_df.shape)
    processed_df, cats = preprocess_data(train_df)
    
    logger.info('Processed train data. Shape: %s', processed_df.shape)
    return processed_df