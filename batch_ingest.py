import s3fs
from s3fs.core import S3FileSystem
import numpy as np
import pickle

import pandas as pd
from pandas_datareader import data as pdr
import yfinance as yfin

def ingest_data():
    # Choose the ticker variables of the stocks the data of which you want to pull
    # Here we are getting 4 years of stock market data from Apple, Google and Amazon
    tickers = ["QQQ", "TECL", "SPX"]
    start_date = '2015-1-1'
    end_date = '2024-1-1'


    yfin.pdr_override()
    # All the data is stored in a pandas dataframe called data
    data = pdr.get_data_yahoo(tickers, start=start_date, end=end_date)


    # s3 = S3FileSystem()
    # # S3 bucket directory
    # DIR = 's3://ece5984-s3-zhenz/Project/'                    # insert your S3 URI here
    # # Push data to S3 bucket as a pickle file
    # with s3.open('{}/{}'.format(DIR, 'data.pkl'), 'wb') as f:
    #     f.write(pickle.dumps(data))

    # Split data by stock code
    # Extract QQQ data
    data_QQQ = data.loc[:, pd.IndexSlice[:, 'QQQ']].copy()
    data_QQQ.columns = ['Adj Close', 'Close', 'High', 'Low', 'Open', 'Volume']

    # Extract TECL data
    data_TECL = data.loc[:, pd.IndexSlice[:, 'TECL']].copy()
    data_TECL.columns = ['Adj Close', 'Close', 'High', 'Low', 'Open', 'Volume']

    # Extract SPX data
    data_SPX = data.loc[:, pd.IndexSlice[:, 'SPX']].copy()
    data_SPX.columns = ['Adj Close', 'Close', 'High', 'Low', 'Open', 'Volume']

    print(data_QQQ)

    # Export each dataset to a pkl file
    data_QQQ.to_pickle("data_QQQ.pkl")
    data_TECL.to_pickle("data_TECL.pkl")
    data_SPX.to_pickle("data_SPX.pkl")

    
    DIR = './'  # Specify the local directory where you want to save the file
    file_path = '{}/{}'.format(DIR, 'data.pkl')

    # Dump data to a local pickle file
    with open(file_path, 'wb') as f:
        pickle.dump(data, f)

ingest_data()