from sqlalchemy import create_engine
from sqlalchemy.exc import OperationalError
import numpy as np
import pandas as pd
from s3fs.core import S3FileSystem


def load_data():
    s3 = S3FileSystem()
    # S3 bucket directory (data warehouse)
    DIR_wh = 's3://ece5984-s3-pyingqi18/Project/Data_Warehouse'

    # Get data from S3 bucket as pickle files
    aapl_train = pd.DataFrame(np.load(s3.open(f"{DIR_wh}/aapl/y_train.pkl"), allow_pickle=True), columns=["Adj Close"])
    aapl_test = pd.DataFrame(np.load(s3.open(f"{DIR_wh}/aapl/y_test.pkl"), allow_pickle=True), columns=["Adj Close"])
    aapl_arima = pd.DataFrame(np.load(s3.open(f"{DIR_wh}/aapl/arima_predictions.pkl"), allow_pickle=True), columns=["Adj Close"])
    aapl_sarima = pd.DataFrame(np.load(s3.open(f"{DIR_wh}/aapl/sarima_predictions.pkl"), allow_pickle=True), columns=["predicted_mean"])

    qqq_train = pd.DataFrame(np.load(s3.open(f"{DIR_wh}/qqq/y_train.pkl"), allow_pickle=True), columns=["Adj Close"])
    qqq_test = pd.DataFrame(np.load(s3.open(f"{DIR_wh}/qqq/y_test.pkl"), allow_pickle=True), columns=["Adj Close"])
    qqq_arima = pd.DataFrame(np.load(s3.open(f"{DIR_wh}/qqq/arima_predictions.pkl"), allow_pickle=True), columns=["Adj Close"])
    qqq_sarima = pd.DataFrame(np.load(s3.open(f"{DIR_wh}/qqq/sarima_predictions.pkl"), allow_pickle=True), columns=["predicted_mean"])

    tecl_train = pd.DataFrame(np.load(s3.open(f"{DIR_wh}/tecl/y_train.pkl"), allow_pickle=True), columns=["Adj Close"])
    tecl_test = pd.DataFrame(np.load(s3.open(f"{DIR_wh}/tecl/y_test.pkl"), allow_pickle=True), columns=["Adj Close"])
    tecl_arima = pd.DataFrame(np.load(s3.open(f"{DIR_wh}/tecl/arima_predictions.pkl"), allow_pickle=True), columns=["Adj Close"])
    tecl_sarima = pd.DataFrame(np.load(s3.open(f"{DIR_wh}/tecl/sarima_predictions.pkl"), allow_pickle=True), columns=["predicted_mean"])

    # Ensure ARIMA and SARIMA align with the dates of y_test
    aapl_arima["Date"] = aapl_test.index
    aapl_sarima["Date"] = aapl_test.index
    qqq_arima["Date"] = qqq_test.index
    qqq_sarima["Date"] = qqq_test.index
    tecl_arima["Date"] = tecl_test.index
    tecl_sarima["Date"] = tecl_test.index

    # Combine y_train + y_test + predictions into full DataFrame
    aapl_combined = pd.concat([
        aapl_train,
        aapl_test.assign(
            Predicted_ARIMA=aapl_arima["Adj Close"].values,
            Predicted_SARIMA=aapl_sarima["predicted_mean"].values
        )
    ])
    qqq_combined = pd.concat([
        qqq_train,
        qqq_test.assign(
            Predicted_ARIMA=qqq_arima["Adj Close"].values,
            Predicted_SARIMA=qqq_sarima["predicted_mean"].values
        )
    ])
    tecl_combined = pd.concat([
        tecl_train,
        tecl_test.assign(
            Predicted_ARIMA=tecl_arima["Adj Close"].values,
            Predicted_SARIMA=tecl_sarima["predicted_mean"].values
        )
    ])

    # Create sqlalchemy engine to connect to MySQL
    user = "admin"
    pw = "9G>c(JolkKDkpanP:jBfp?P+<{yJ"
    endpnt = "data-eng-db.cluster-cwgvgleixj0c.us-east-1.rds.amazonaws.com"
    db_name = "pyingqi18"

    # Connect to MySQL server (without specifying a database)
    engine = create_engine(f"mysql+pymysql://{user}:{pw}@{endpnt}")

    # Check if the database already exists
    with engine.connect() as connection:
        db_exists = connection.execute(f"SHOW DATABASES LIKE '{db_name}';").fetchone()
        if not db_exists:
            # Create the database if it does not exist
            connection.execute(f"CREATE DATABASE {db_name}")

    # Reconnect to the specific database
    engine = create_engine(f"mysql+pymysql://{user}:{pw}@{endpnt}/{db_name}")

    # Insert DataFrames into MySQL DB
    aapl_combined.to_sql('aapl_combined', con=engine, if_exists='replace', chunksize=1000)
    qqq_combined.to_sql('qqq_combined', con=engine, if_exists='replace', chunksize=1000)
    tecl_combined.to_sql('tecl_combined', con=engine, if_exists='replace', chunksize=1000)
