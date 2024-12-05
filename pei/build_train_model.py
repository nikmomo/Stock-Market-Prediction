import s3fs
from s3fs.core import S3FileSystem
import numpy as np
import pickle
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
from keras.layers import LSTM, Dense, Input, MultiHeadAttention, LayerNormalization
from keras.models import Sequential, Model
from sklearn.preprocessing import MinMaxScaler
import tempfile


def build_train():
    s3 = S3FileSystem()

    # S3 bucket directories
    datasets = {
        "qqq": "s3://ece5984-s3-pyingqi18/Project/Data_Warehouse/qqq",
        "tecl": "s3://ece5984-s3-pyingqi18/Project/Data_Warehouse/tecl",
        "aapl": "s3://ece5984-s3-pyingqi18/Project/Data_Warehouse/aapl"
    }

    # Common parameters
    window_size = 60  # Sliding window size for LSTM and Transformer
    epochs = 25
    batch_size = 8

    for name, dir_path in datasets.items():
        # Load the processed data
        y_train = np.load(s3.open(f"{dir_path}/y_train.pkl"), allow_pickle=True)
        y_test = np.load(s3.open(f"{dir_path}/y_test.pkl"), allow_pickle=True)

        # Normalize data
        scaler = MinMaxScaler()
        y_train_scaled = scaler.fit_transform(y_train.reshape(-1, 1))
        y_test_scaled = scaler.transform(y_test.reshape(-1, 1))

        # Prepare data for LSTM and Transformer
        def prepare_lstm_data(y, window_size):
            X, y_windowed = [], []
            for i in range(len(y) - window_size):
                X.append(y[i:i + window_size])
                y_windowed.append(y[i + window_size])
            return np.array(X), np.array(y_windowed)

        X_train_lstm, y_train_lstm = prepare_lstm_data(y_train_scaled, window_size)
        X_test_lstm, y_test_lstm = prepare_lstm_data(y_test_scaled, window_size)

        # ARIMA Model
        arima_model = ARIMA(y_train, order=(5, 1, 0))
        arima_fitted = arima_model.fit()
        arima_predictions = arima_fitted.forecast(steps=len(y_test))

        # Save ARIMA predictions
        with s3.open(f"{dir_path}/arima_predictions.pkl", 'wb') as f:
            pickle.dump(arima_predictions, f)

        # LSTM Model
        lstm_model = Sequential([
            LSTM(32, input_shape=(window_size, 1), activation='relu'),
            Dense(1)
        ])
        lstm_model.compile(optimizer='adam', loss='mean_squared_error')
        lstm_model.fit(X_train_lstm, y_train_lstm, epochs=epochs, batch_size=batch_size,
                       validation_data=(X_test_lstm, y_test_lstm), verbose=1)

        # Save LSTM model
        with tempfile.TemporaryDirectory() as tempdir:
            lstm_model.save(f"{tempdir}/lstm_{name}.h5")
            s3.put(f"{tempdir}/lstm_{name}.h5", f"{dir_path}/lstm_{name}.h5")

        # Transformer Model
        input_layer = Input(shape=(window_size, 1))
        attention = MultiHeadAttention(num_heads=4, key_dim=2)(input_layer, input_layer)
        norm = LayerNormalization(epsilon=1e-6)(attention)
        dense = Dense(1)(norm)
        transformer_model = Model(inputs=input_layer, outputs=dense)
        transformer_model.compile(optimizer='adam', loss='mean_squared_error')
        transformer_model.fit(X_train_lstm, y_train_lstm, epochs=epochs, batch_size=batch_size,
                              validation_data=(X_test_lstm, y_test_lstm), verbose=1)

        # Save Transformer model
        with tempfile.TemporaryDirectory() as tempdir:
            transformer_model.save(f"{tempdir}/transformer_{name}.h5")
            s3.put(f"{tempdir}/transformer_{name}.h5", f"{dir_path}/transformer_{name}.h5")

