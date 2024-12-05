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
    tickers = ["QQQ", "TECL", "AAPL"]
    start_date = '2015-1-1'
    end_date = '2024-1-1'


    yfin.pdr_override()
    # All the data is stored in a pandas dataframe called data
    stock_data = pdr.get_data_yahoo(tickers, start=start_date, end=end_date)

    # Initialize S3 file system and specify the S3 bucket directory
    s3 = s3fs.S3FileSystem()
    s3_dir = 's3://ece5984-s3-pyingqi18/Project/Data_Lake'  # Insert  your S3 bucket address here. Create a directory as Lab2/batch_ingest

    # Save the modified dataset to the S3 bucket as a pickle file
    with s3.open(f'{s3_dir}/data.pkl', 'wb') as file:
        pickle.dump(stock_data, file)