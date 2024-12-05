import s3fs
from s3fs.core import S3FileSystem
import numpy as np
import pickle
import pandas as pd


def feature_extract():
    s3 = S3FileSystem()

    # S3 bucket directory (data warehouse)
    DIR_wh = 's3://ece5984-s3-pyingqi18/Project/Data_Warehouse'

    # Get data from S3 bucket as pickle files
    qqq_df = np.load(s3.open(f'{DIR_wh}/clean_qqq.pkl'), allow_pickle=True)
    tecl_df = np.load(s3.open(f'{DIR_wh}/clean_tecl.pkl'), allow_pickle=True)
    aapl_df = np.load(s3.open(f'{DIR_wh}/clean_aapl.pkl'), allow_pickle=True)

    def split_data(dataframe, name):
        # Extract target variable (Adj Close)
        target = dataframe['Adj Close']

        # Calculate split index for 90% train and 10% test
        split_index = int(len(target) * 0.9)

        # Split into train and test sets
        y_train = target.iloc[:split_index]
        y_test = target.iloc[split_index:]


        # Push extracted features to data warehouse
        DIR_out = f's3://ece5984-s3-pyingqi18/Project/Data_Warehouse/{name}'
        with s3.open(f'{DIR_out}/y_train.pkl', 'wb') as f:
            f.write(pickle.dumps(y_train))
        with s3.open(f'{DIR_out}/y_test.pkl', 'wb') as f:
            f.write(pickle.dumps(y_test))

    # Apply the function to each dataset
    split_data(qqq_df, 'qqq')
    split_data(tecl_df, 'tecl')
    split_data(aapl_df, 'aapl')





