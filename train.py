from datetime import datetime
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras import callbacks
import pandas as pd
import os
from typing import Dict, List, Union
import wandb
from wandb.keras import WandbCallback
from models.network import init_model
import csv


def init_training(
    config: dict,
    data: Dict[str, str],
    data_root: str = "data"
) -> None:
    print("Initialize training script.")
    for data_horizon, data_set in data.items():
        # Init wheigt and biases for logging
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        experiment_name = str(now) + "_" + data_horizon
        wandb.init(
            project="stock-returns-with-option-data",
            name=experiment_name,
            entity="realptkkit",
            tags=[data_horizon]
            )
        wandb.config = {
            "name": experiment_name,
            "date": now,
            "dataset": data_horizon,
            "learning_rate": 0.001,
            "epochs": 20,
            "batch_size": 128
        }

        # If not received, load data.
        if type(data_set) == str:
            path = os.path.join(
                data_root, "preprocessed",
                data_horizon + ".csv"
            )
            data = pd.read_csv(path, index_col="date", parse_dates=["date"])

        # Create dataframes for features and target
        target = data["target_eret"]
        theory_target = data["theory_eret"]
        features = data.drop(['secid', 'theory_eret', "target_eret"], axis=1)

        # split into train, validation and test data
        X_train, X_test, y_train, y_test = train_test_split(
            features, target, test_size=0.2, random_state=1
        )

        # get_train_split(data, time_horizon)
        model = train(experiment_name, data_horizon, config, X_train, y_train)
        evaluation(experiment_name, data_horizon, model, config, X_test, y_test)


def get_train_split(data, time_horizon):
    pass


def train(experiment_name: str, data_horizon: str, config: dict, X_train, y_train) -> str:
    # Stup config
    print("Start Training Script")
    print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
    train_config = config["train"]
    model_config = config["model"]

    # Initiate model and train
    inputs = X_train.shape[1]
    model_config["input_dim"] = inputs
    model = init_model(data_horizon, model_config)
    checkpoint_filepath = train_config["checkpoint_path"]
    # model_checkpoint_callback = callbacks.ModelCheckpoint(
    #     filepath=checkpoint_filepath,
    #     save_best_only=True,
    #     save_weights_only=False,
    #     monitor="loss"
    # )
    early_stopping_callback = callbacks.EarlyStopping(
        monitor="loss",
        patience=3,
        restore_best_weights=True
    )
    history = model.fit(
        X_train.to_numpy(),
        y_train.to_numpy(),
        epochs=20,
        batch_size=256,
        validation_split=0.2,
        callbacks=[
            WandbCallback(), early_stopping_callback
        ]
    )
    path = os.path.join(checkpoint_filepath, experiment_name)
    model.save(path)
    return model


def evaluation(experiment_name: str, data_horizon: str, model, config: dict, X_test, y_test):
    prediction = model.predict(X_test)

    # Evaluation
    mae = tf.metrics.mean_absolute_error(
        y_true=y_test,
        y_pred=prediction.squeeze()
    ).numpy()
    mse = tf.metrics.mean_squared_error(
        y_true=y_test,
        y_pred=prediction.squeeze()
    ).numpy()
    print(f" ------ Evaluation Results ------ \n MAE: {mae} \n MSE: {mse}")
    # Date, Dataset, MAE, MSE
    fields = [experiment_name, data_horizon, mae, mse]
    path = config["evaluation"]["evaluation_csv"]
    with open(path, 'a') as f:
        writer = csv.writer(f)
        writer.writerow(fields)
