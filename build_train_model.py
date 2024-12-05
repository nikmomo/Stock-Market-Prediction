import s3fs
from s3fs.core import S3FileSystem
import numpy as np
import pickle
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from pmdarima import auto_arima
import tempfile


def build_train():
    s3 = S3FileSystem()

    # S3 bucket directories
    datasets = {
        "qqq": "s3://ece5984-s3-pyingqi18/Project/Data_Warehouse/qqq",
        "tecl": "s3://ece5984-s3-pyingqi18/Project/Data_Warehouse/tecl",
        "aapl": "s3://ece5984-s3-pyingqi18/Project/Data_Warehouse/aapl"
    }

    for name, dir_path in datasets.items():
        # Load the processed data
        y_train = np.load(s3.open(f"{dir_path}/y_train.pkl"), allow_pickle=True)
        y_test = np.load(s3.open(f"{dir_path}/y_test.pkl"), allow_pickle=True)

        # Perform differencing to make the series stationary
        y_train_diff = np.diff(y_train, n=1)
        y_test_diff = np.diff(y_test, n=1)

        # Restore original scale of predictions after differencing
        def restore_diff(original_series, diff_series):
            restored_series = []
            last_value = original_series[0]
            for diff in diff_series:
                restored_series.append(last_value + diff)
                last_value += diff
            return np.array(restored_series)

        # **ARIMA Model with Differencing**
        # Automatically determine the best (p, d, q)
        auto_arima_model = auto_arima(y_train_diff, seasonal=False, trace=True, error_action="ignore", suppress_warnings=True)
        p, d, q = auto_arima_model.order
        print(f"Best ARIMA Order for {name}: (p={p}, d={d}, q={q})")

        # Fit ARIMA Model
        arima_model = ARIMA(y_train_diff, order=(p, d, q))
        arima_fitted = arima_model.fit()
        arima_predictions_diff = arima_fitted.forecast(steps=len(y_test_diff))

        # Restore ARIMA predictions to the original scale
        arima_predictions = restore_diff(y_train[-1:], arima_predictions_diff)

        # **Check and Align Length**
        # If arima_predictions length is less than y_test, pad the remaining with the last value
        if len(arima_predictions) < len(y_test):
            padding_length = len(y_test) - len(arima_predictions)
            arima_predictions = np.append(
                arima_predictions, [arima_predictions[-1]] * padding_length
            )
            print(f"Padding ARIMA predictions for {name}: Added {padding_length} points.")

        # Save ARIMA predictions
        with s3.open(f"{dir_path}/arima_predictions.pkl", 'wb') as f:
            pickle.dump(arima_predictions, f)


        # **SARIMA Model**
        # Automatically determine (P, D, Q, m) with auto_arima
        auto_sarima_model = auto_arima(
            y_train,  # SARIMA directly uses original data
            seasonal=True,
            m=12,  # Assumes monthly seasonality; adjust if needed
            trace=True,
            error_action="ignore",
            suppress_warnings=True,
            stepwise=True
        )
        P, D, Q, m = auto_sarima_model.seasonal_order
        print(f"Best SARIMA Order for {name}: Seasonal (P={P}, D={D}, Q={Q}, m={m})")

        # Fit SARIMA Model
        sarima_model = SARIMAX(
            y_train,
            order=(p, d, q),
            seasonal_order=(P, D, Q, m)
        )
        sarima_fitted = sarima_model.fit(disp=False)
        sarima_predictions = sarima_fitted.forecast(steps=len(y_test))

        # **Check and Align Length for SARIMA**
        if len(sarima_predictions) < len(y_test):
            padding_length = len(y_test) - len(sarima_predictions)
            sarima_predictions = np.append(sarima_predictions, [sarima_predictions[-1]] * padding_length)
            print(f"Padding SARIMA predictions for {name}: Added {padding_length} points.")

        # Save SARIMA predictions
        with s3.open(f"{dir_path}/sarima_predictions.pkl", 'wb') as f:
            pickle.dump(sarima_predictions, f)
