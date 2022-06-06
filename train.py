from sklearn.model_selection import train_test_split
import tensorflow as tf
import pandas as pd
import os
from typing import Dict, List, Union

from models.network import create_model


def init_training(data: Dict[str, str], data_root: str = "data") -> None:
    for data_horizon, data_set in data:
        if type(data_set) == str:
            path = os.path.join(data_root, "preprocessed", data_horizon + ".csv")
            data = pd.read_csv(path, index_col="date", parse_dates=["date"])
        train(data)


def train(data: pd.DataFrame) -> str:

    # Create dataframes for features and target
    target = data["target_eret"]
    theory_target = data["theory_eret"]
    features = data.drop(['secid', 'theory_eret', "target_eret"], axis=1)

    # split into train, validation and test data
    X_train, X_test, y_train, y_test = train_test_split(
        features, target, test_size=0.2, random_state=1)

    tf.random.set_seed(42)
    input_dim = features.shape[1]
    model = create_model(input_dim=input_dim) 
    model.fit(X_train, y_train, epochs=100, verbose=0, batch_size=32, validation_split=0.2)

    preds_6 = model.predict(X_test)

    # Evaluation
    mae_6 = tf.metrics.mean_absolute_error(y_true=y_test,
                                           y_pred=preds_6.squeeze()).numpy()
    mse_6 = tf.metrics.mean_squared_error(y_true=y_test,
                                          y_pred=preds_6.squeeze()).numpy()

    print(mae_6, mse_6)
