from typing import Dict, List, Union
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn.preprocessing import StandardScaler as SSC
from tensorflow.keras import Model, models


def preprocessing(data: Dict[str, str], data_root: str = "data") -> Dict[str, pd.DataFrame]:
    # dict_data_cleaned = clean_data(data, data_root)
    # dict_data_preprocessed = standardization(data, data_root)
    dimension_reduction(data, data_root)


def clean_data(
        data: Dict[str, Union[str, pd.DataFrame]],
        data_root: str = "data"
) -> Dict[str, pd.DataFrame]:

    print("Start cleaning the data")
    data_horizon = str
    data_set = pd.DataFrame
    data_cleaned = data
    for data_horizon, data_set in data.items():
        df = data_set
        if type(data_set) == str:
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
        cleaned_path = os.path.join(
            data_root, "cleaned", data_horizon + ".csv")
        df.to_csv(cleaned_path)
        data_cleaned[data_horizon] = df
        print(f"Cleaned {data_horizon} and saved to {cleaned_path}")
    return data_cleaned


def standardization(
        data: Dict[str, Union[str, pd.DataFrame]],
        data_root: str = "data") -> Dict[str, pd.DataFrame]:

    scaler = SSC()
    data_preprocessed = data
    for data_horizon, data_set in data.items():
        df = data_set
        if type(data_set) == str:
            df = pd.read_csv(os.path.join(data_root, "cleaned", data_horizon + ".csv"),
                             index_col="date", parse_dates=["date"])
        df_dropped = df.drop(['secid', 'theory_eret', "target_eret"], axis=1)
        columns = df_dropped.columns.tolist()
        data_scaled = scaler.fit_transform(df_dropped)
        X_scaled = pd.DataFrame(
            data_scaled, index=df_dropped.index, columns=columns)
        X_scaled['secid'], X_scaled['theory_eret'], X_scaled['target_eret'] = [
            df['secid'], df['theory_eret'], df['target_eret']]
        final_path = os.path.join(
            data_root, "preprocessed", data_horizon + ".csv")
        X_scaled.to_csv(final_path)
        data_preprocessed[data_horizon] = X_scaled
        print(f"Preprocessed {data_horizon} and saved to {final_path}")
    return data_preprocessed


def dimension_reduction(
        data_dict: Dict[str, Union[str, pd.DataFrame]],
        data_root: str = "data",
        model_root: str = "models/autoencoder") -> Dict[str, pd.DataFrame]:

    data = pd.DataFrame
    encoder = Model
    for data_horizon, data_set in data_dict.items():
        if type(data_set) == str:
            print("Loading preprocessed data: ", data_horizon)
            path = os.path.join(
                data_root, "preprocessed",
                data_horizon + ".csv"
            )
            data = pd.read_csv(
                path, parse_dates=["date"]
            )
        data.sort_values(by="date", inplace=True)
        features = data.drop(["date", 'secid', 'theory_eret', "target_eret"], axis=1)
        data.drop(columns=features.columns, inplace=True)

        encoder = models.load_model(os.path.join(model_root, f"autoencoder_{data_horizon}"))

        features_encoded = encoder.predict(features)
        columns = []
        for i in range(0, features_encoded.shape[1]):
            columns.append(f"encoded_{i}")
        features_encoded_frame = pd.DataFrame(features_encoded, columns=columns)
        data_encoded = pd.concat([data, features_encoded_frame], axis=1)
        data_encoded.set_index("date", inplace=True)
        data_encoded.sort_values(by="date", inplace=True)

        final_path = os.path.join(data_root, "preprocessed", data_horizon + "_encoded.csv")
        data_encoded.to_csv(final_path)
