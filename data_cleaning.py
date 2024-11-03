import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import sys
import os

# Load the original data from command line argument
if len(sys.argv) < 2:
    print("Usage: python script.py <pkl_file>")
    sys.exit(1)

input_file = sys.argv[1]
data = pd.read_pickle(input_file)

file_name = os.path.splitext(os.path.basename(input_file))[0]

# Perform log transformation, 1.5IQR outlier removal, and linear interpolation on each dataset
def preprocess_data(data):

    data = data.sort_index()

    # Log transformation (adding a small constant to avoid log(0))
    log_data = data.apply(lambda x: np.log(x + 1e-5) if np.issubdtype(x.dtype, np.number) else x)

    # 1.5 IQR outlier removal
    def remove_outliers(df):
        for column in df.select_dtypes(include=[np.number]).columns:
            Q1 = df[column].quantile(0.25)
            Q3 = df[column].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            df[column] = df[column].apply(lambda x: x if lower_bound <= x <= upper_bound else np.nan)
        return df

    outlier_removed_data = remove_outliers(log_data)

    # Linear interpolation to fill missing values
    interpolated_data = outlier_removed_data.interpolate(method='linear')

    # Forward fill and backward fill to handle any remaining NaN values, especially at the first row
    interpolated_data = interpolated_data.fillna(method='bfill').fillna(method='ffill')

    return interpolated_data

processed_data = preprocess_data(data)

print(processed_data)

processed_data.to_pickle(f"./data/{file_name}_processed.pkl")