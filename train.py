from datetime import datetime
from re import A
from sqlite3 import Timestamp
from attr import field
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras import callbacks, Model, backend
import pandas as pd
import os
from typing import Dict, List, Tuple, Union
import wandb
from wandb.keras import WandbCallback
from models.network import init_model
import csv
from tensorflow.keras.metrics import mean_absolute_error, mean_squared_error
import numpy as np
from utils.log_utils import average_yearly_logs
import gc
from utils.oos_metric import r_squared
from utils.window_generator import WindowGenerator


def init_training(
    config: dict,
    data_dict: dict,
    data_root: str = "data",
    ensamble_num: int = 10,
) -> None:

    # Initialize training script and configs
    print("Initialize training script.")
    gpus = tf.config.experimental.list_physical_devices('GPU')
    tf.config.experimental.set_memory_growth(gpus[0], True)
    print("TF_GPU_ALLOCATOR: ", os.getenv("TF_GPU_ALLOCATOR"))

    train_config = config["train"]
    model_config = config["model"]
    evaluation_config = config["evaluation"]
    eval_metrics = evaluation_config["model_log_metrics"]

    for data_horizon, data_set in data_dict.items():

        # Initialize stuff for ensamble run
        print("Started run for data_set: ", data_horizon)
        data = pd.DataFrame
        ensamble_num = train_config["ensamble_num"]
        now = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        model_log_list = []

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
        combined_target = ["target_eret", "theory_eret"]
        features = data.drop(
            ['secid', 'theory_eret', "target_eret"], axis=1).columns

        # Create WindowGenerate for dataloading
        train_w = evaluation_config["training_width"]
        val_w = evaluation_config["validation_width"]
        test_w = evaluation_config["testing_width"]
        wg = WindowGenerator(
            data=data,
            labels=combined_target,
            features=features,
            training_width=train_w,
            validation_width=val_w,
            testing_width=test_w
        )

        # Ensamble Training
        for j in range(ensamble_num):

            # Name of experiment
            print(f"START Experiment Nr. {j}/{ensamble_num}")
            experiment_name = f"{str(now)}_{data_horizon}_ensamble_{str(j)}"

            # Init wheigt and biases for logging
            wandb_config = {
                "name": experiment_name,
                "date": now,
                "dataset": data_horizon,
                "model_config": model_config,
                "epochs": train_config["epochs"],
                "batch_size": train_config["batch_size"]
            }

            wandb.init(
                project="stock-returns-with-option-data",
                name=experiment_name,
                entity="realptkkit",
                tags=[data_horizon],
                config=wandb_config
            )

            # Initiate the network and train the model
            model, model_log = train(experiment_name, config, wg)

            # Save only one model per run
            model_save_path = train_config["model_save_path"]
            path = os.path.join(model_save_path, experiment_name)
            if j == 0:
                model.save(path)

            # Append model log to list
            model_log_list.append((j, model_log))
            wandb.finish(
                quiet=True
            )
            print(f"FINISHED Experiment Nr. {j}/{ensamble_num}")

        # Log model results and create csv for each run
        evaluation_range = wg.get_evaluation_range()
        dir = evaluation_config["results_root"]
        path = os.path.join(dir, f"evaluation_run_{now}_{data_horizon}_{ensamble_num}.csv")
        averaged_yearly_logs = average_yearly_logs(
            model_log_list, eval_metrics, evaluation_range
        )
        averaged_yearly_logs.to_csv(path, index=True)

        # Log ensamble results for each run in a single csv
        print("Logging ensamble results to: ", path)
        path = evaluation_config["ensamble_log_csv"]
        ensamble_log_col = [
            "ensamble_name",
            "data_horizon",
            "ensamble_num",
            "train-val-test_width"
        ] + eval_metrics
        fields = {
            "ensamble_name": f"ensamble_{now}_{data_horizon}_SW",
            "data_horizon": data_horizon,
            "ensamble_num": ensamble_num,
            "train-val-test_width": f"{train_w}-{val_w}-{test_w}"

        }
        for metric in eval_metrics:
            fields[metric] = averaged_yearly_logs[metric].mean()

        with open(path, 'a', newline='\n') as f:
            writer = csv.DictWriter(f, fieldnames=ensamble_log_col)
            writer.writerow(fields)
            f.close()


def get_train_split(data, time_horizon):
    pass


def train(experiment_name: str, config: dict, wg: WindowGenerator) -> str:
    # Setup config
    print("Start Training Script")
    print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

    train_config = config["train"]
    model_config = config["model"]
    evaluation_config = config["evaluation"]
    model = Model

    # Initiate model and train
    model = init_model(experiment_name, model_config)
    early_stopping_callback = callbacks.EarlyStopping(
        monitor="r_squared_fn",
        patience=3,
        mode="max",
        restore_best_weights=True
    )

    # Prepare logging
    model_log_metrics = evaluation_config["model_log_metrics"]
    model_log = pd.DataFrame(columns=model_log_metrics)

    # config tensorflow session
    while not wg.data_end_reached():
        backend.clear_session()
        # Get current testing year
        log_dict = {}
        current_year = wg.get_current_evaluation_year()
        print("Training and Validation window currently until: ", current_year)
        log_dict["evaluation_year"] = current_year
        # Split into train, validation and test data
        X_train, y_train_comb, X_val, y_val_combined, X_test, y_test_comb = wg.split()

        y_train = y_train_comb[:, 0]  # 0 == "target_eret"
        y_test = y_test_comb[:, 0]
        y_val = y_val_combined[:, 0]
        predictions_theory = y_test_comb[:, 1]  # 1 == theory_eret

        y_val_combined = None
        y_test_comb = None

        # Train Model
        model.fit(
            X_train,
            y_train,
            epochs=train_config["epochs"],
            batch_size=train_config["batch_size"],
            validation_data=(X_val, y_val),
            callbacks=[
                WandbCallback(), early_stopping_callback
            ]
        )
        
        # Evaluate
        prediction = model.predict(X_test)
        _, eval_metrics_filled = evaluation(
            prediction, y_test, predictions_theory)

        # Log results
        log_dict = log_dict | eval_metrics_filled
        log_df = pd.DataFrame([log_dict])
        model_log = pd.concat([model_log, log_df])

        # Free some memory
        prediction = predictions_theory = y_val = y_test = y_train = None
        eval_metrics_filled = None
        log_dict = None
        X_train = y_train_comb = X_val = y_val_combined = X_test = y_test_comb = None
        
        # Update sliding window
        wg.update_dates()
        gc.collect()

    # Convert year to int and set as index
    model_log["evaluation_year"] = model_log["evaluation_year"].astype(int)
    model_log.set_index("evaluation_year", inplace=True)

    return model, model_log


def evaluation(
    prediction: np.array,
    y_test: pd.DataFrame,
    predictions_theory: pd.DataFrame
) -> Tuple[np.array, dict]:

    oss, rss, tss = r_squared(
        target_returns=y_test,
        predictions=prediction.squeeze()
    )
    oss_theory, _, _ = r_squared(
        target_returns=y_test,
        predictions=predictions_theory
    )
    # Evaluation
    mae = mean_absolute_error(
        y_true=y_test,
        y_pred=prediction.squeeze()
    ).numpy()
    mse = mean_squared_error(
        y_true=y_test,
        y_pred=prediction.squeeze()
    ).numpy()
    print(f""" ------ Evaluation Results ------
    MAE: {mae} | MSE: {mse}
    OSS: {oss} | RSS: {rss} | TSS: {tss}
    OSS_theory: {oss_theory}
    """)

    eval_metrics = {
        "MAE": mae,
        "MSE": mse,
        "OSS": oss,
        "RSS": rss,
        "TSS": tss,
        "OSS_theory": oss_theory
    }
    gc.collect()
    return prediction, eval_metrics
