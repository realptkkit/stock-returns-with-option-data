from typing import Dict, List
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn.preprocessing import StandardScaler as SSC


def preprocessing(data_root: str, data: Dict[str, str]) -> str:
    # clean_data(data_root, data)
    standardization(data_root, data)


def clean_data(data_root: str, data: Dict[str, str]) -> None:
    print("Start cleaning the data")
    data_horizon = str
    data_set = pd.DataFrame
    for data_horizon, data_set in data.items():
        df = pd.read_csv(os.path.join(data_root, "raw", data_set),
                         index_col="date", parse_dates=["date"])
        print(data_horizon, ": ", df.shape)
        df.drop(df.columns[0], axis=1, inplace=True)

        col_miss = []
        for col in df.columns:
            if df[col].isnull().sum():
                col_miss.append(col)
                print("Sum of Null Values for ", col,
                      ": ", df[col].isnull().sum())

        df.interpolate(method="linear", axis=1, inplace=True)
        cleaned_path = os.path.join(data_root, "cleaned", data_horizon + ".csv")
        df.to_csv(cleaned_path)
        print(f"Cleaned {data_horizon} and saved to {cleaned_path}")



def standardization(data_root, data: Dict[str, str]) -> None:
    scaler = SSC()
    for data_horizon, data_set in data.items():
        df = pd.read_csv(os.path.join(data_root, "cleaned", data_horizon + ".csv"),
                         index_col="date", parse_dates=["date"])
        df.drop(['secid', 'theory_eret', "target_eret"],
                axis=1, inplace=True)
        columns = df.columns.tolist()
        data_scaled = scaler.fit_transform(df)
        X_scaled = pd.DataFrame(data_scaled, index=df.index, columns=columns)
        final_path = os.path.join(data_root, "preprocessed", data_horizon + ".csv")
        X_scaled.to_csv(final_path)
        print(f"Preprocessed {data_horizon} and saved to {final_path}")