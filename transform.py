import s3fs
from s3fs.core import S3FileSystem
import numpy as np
import pickle
import pandas as pd

# Outlier removal function, 1.5 IQR technique
def remove_outliers(df):
    for column in df.select_dtypes(include=['number']).columns:
        Q1 = df[column].quantile(0.25)
        Q3 = df[column].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        df[column] = df[column].apply(lambda x: x if lower_bound <= x <= upper_bound else np.nan)
    return df

# Preprocessing function
def preprocess_dataframe(df):
    df = df.dropna()  # drop NaN
    df = df.apply(lambda x: np.log(x + 1e-5) if np.issubdtype(x.dtype, np.number) else x) # log transformation
    df = remove_outliers(df) # remove outliers
    df = df.interpolate(method='time').fillna(method='bfill').fillna(method='ffill') # interpolate removed missing values
    return df


# Perform log transformation, 1.5IQR outlier removal, and linear interpolation on each dataset
def transform_data():
    s3 = S3FileSystem()
    # S3 bucket directory (data lake)
    DIR = 's3://ece5984-s3-pyingqi18/Project/Data_Lake'  # Insert your S3 bucket address here. Read from the directory you created in batch ingest: Lab2/batch_ingest/
    # Get data from S3 bucket as a pickle file
    raw_data = np.load(s3.open('{}/{}'.format(DIR, 'data.pkl')), allow_pickle=True)

    # Dividing the raw dataset for each company individual company
    raw_data.columns = raw_data.columns.swaplevel(0, 1)
    raw_data.sort_index(axis=1, level=0, inplace=True)
    df_qqq_rw = raw_data['QQQ']
    df_tecl_rw = raw_data['TECL']
    df_aapl_rw = raw_data['AAPL']

    # Preprocess individual datasets
    df_qqq = preprocess_dataframe(df_qqq_rw)
    df_tecl = preprocess_dataframe(df_tecl_rw)
    df_aapl = preprocess_dataframe(df_aapl_rw)


# Push cleaned data to S3 bucket warehouse
    DIR_wh = 's3://ece5984-s3-pyingqi18/Project/Data_Warehouse'   # Insert your S3 bucket address here. Create a directory as: Lab2/transformed/
    with s3.open('{}/{}'.format(DIR_wh, 'clean_qqq.pkl'), 'wb') as f:
        f.write(pickle.dumps(df_qqq))
    with s3.open('{}/{}'.format(DIR_wh, 'clean_tecl.pkl'), 'wb') as f:
        f.write(pickle.dumps(df_tecl))
    with s3.open('{}/{}'.format(DIR_wh, 'clean_aapl.pkl'), 'wb') as f:
        f.write(pickle.dumps(df_aapl))