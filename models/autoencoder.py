import gc
from typing import Dict, List, Union
import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.keras import layers, Model, Input, callbacks, backend
import os
import json


def train_autoenoder(
        config_path: str = "config.json",
        model_root: str = "models/autoencoder",
        epochs: int = 200) -> Model:
    """Trains the autoencoder and saves the encoder part.add()

    Args:
        config_path (str, optional): Path to the config file. Defaults to "config.json".
        model_root (str, optional): Folder where model should be stored. 
        Defaults to "models/autoencoder".
        epochs (int, optional): Amount of epochs for the training. Defaults to 200.

    Returns:
        Model: _description_
    """
    config = Dict
    with open(config_path, "r") as json_file:
        config = json.load(json_file)
        print("Loaded config file")
    data_dict = config["data"]
    data_root = config["settings"]["data_root"]

    for data_horizon, data_set in data_dict.items():
        backend.clear_session()
        gc.collect()
        # If not received, load data and sort ascending.
        if type(data_set) == str:
            print("Loading preprocessed data: ", data_horizon)
            path = os.path.join(
                data_root, "preprocessed",
                data_horizon + ".csv"
            )
            data = pd.read_csv(
                path, index_col="date",
                parse_dates=["date"]
            )
        data.sort_values(by="date", inplace=True)

        # Extract Label and Feature column names including theory
        features = data.drop(
            ['secid', 'theory_eret', "target_eret"], axis=1)
        data = None
        X_train, X_test, _, _ = train_test_split(
            features, features, test_size=0.2, random_state=1
        )
        features = None
        early_stopping_callback = callbacks.EarlyStopping(
            monitor="val_mse",
            patience=8,
            mode="min",
            restore_best_weights=True
        )
        model = autoencoder = Model
        model, autoencoder = create_autoencoder()
        model.fit(
            X_train.to_numpy(),
            X_train.to_numpy(),
            epochs=epochs,
            batch_size=256,
            validation_data=(X_test.to_numpy(), X_test.to_numpy()),
            callbacks=[early_stopping_callback]
        )
        model = None
        X_train = X_test = None
        autoencoder.save(os.path.join(
            model_root, f"autoencoder_{data_horizon}"))
        backend.clear_session()
        gc.collect()


def create_autoencoder(
        input_dim: int = 935,
        loss: str = "mse",
        optimizer: str = "adam",
        metrics: List[str] = ["mse", "mae"],
        bottleneck_num: int = 15
) -> Model:
    """Creates the autoencoder model"""
    # Encoder
    inputs = Input(shape=(input_dim,), name="autoencoder")
    x = layers.Dense(
        input_dim/2,
        activation='relu',
        kernel_initializer='he_uniform',
        name=f"Encoder_0"
    )(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Dense(
        input_dim/4,
        activation='relu',
        kernel_initializer='he_uniform',
        name=f"Encoder_1"
    )(x)
    x = layers.BatchNormalization()(x)
    # Bottleneck
    bottleneck = layers.Dense(
        bottleneck_num,
        activation='relu',
        kernel_initializer='he_uniform',
        name=f"Bottleneck"
    )(x)
    # Decoder
    x = layers.Dense(
        input_dim/4,
        activation='relu',
        kernel_initializer='he_uniform',
        name=f"Decoder_0"
    )(bottleneck)
    x = layers.BatchNormalization()(x)
    x = layers.Dense(
        input_dim/2,
        activation='relu',
        kernel_initializer='he_uniform',
        name=f"Decoder_1"
    )(x)
    x = layers.BatchNormalization()(x)
    outputs = layers.Dense(
        input_dim,
        activation="linear",
        kernel_initializer='he_uniform',
        name="outputs"
    )(x)

    model = Model(
        inputs=inputs,
        outputs=outputs,
        name=f"Autoencoder"
    )
    model.compile(
        loss=loss,  # Mean Squared Error
        optimizer=optimizer,  # Adam Optimizer
        metrics=metrics  # A variety of error functions
    )

    autoencoder = Model(
        inputs,
        bottleneck,
    )

    return model, autoencoder


if __name__ == "__main__":
    train_autoenoder()
