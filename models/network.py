from typing import Dict, List
import tensorflow as tf
from tensorflow.keras import layers, losses, optimizers, Model, Input, callbacks
from keras.layers import Dropout


def init_model(data_horizon: str, config: dict) -> Model:
    input_dim = config["input_dim"]
    hidden_layer = config["hidden_layer"]
    num_units = config["num_units"]
    loss = config["loss"]
    optimizer = config["optimizer"]
    metrics = config["metrics"]
    learning_rate = config["learning_rate"]
    model = create_model(
        input_dim=input_dim,
        hidden_layer=hidden_layer,
        num_units=num_units,
        loss=loss,
        optimizer=optimizer,
        metrics=metrics,
        learning_rate=learning_rate,
        data_horizon=data_horizon
    )
    return model


def create_model(
        data_horizon: str,
        input_dim: int = 938,
        hidden_layer: int = 5,
        num_units: int = 600,
        loss: str = "mse",
        optimizer: str = "adam",
        metrics: List[str] = ["mse", "mape", "mae", "msle"],
        learning_rate: float = 0.001,
) -> Model:
    inputs = Input(shape=(input_dim,), name="option_data")
    x = layers.Dense(
        num_units,
        activation='relu',
        kernel_initializer='he_uniform',
        name=f"dense_0"
    )(inputs)
    x = Dropout(0.4)(x)
    print(f"dense_0 layer has {num_units} units")
    for i in range(hidden_layer-1):
        delta = (i+1)*(num_units/hidden_layer) + 50
        units = num_units-delta
        x = layers.Dense(
            units,
            activation="relu",
            kernel_initializer='he_uniform',
            name=f"dense_{i+1}"
        )(x)
        x = Dropout(0.3)(x)
        print(f"dense_{i+1} layer has {units} units")
    outputs = layers.Dense(
        1,
        activation="linear",
        kernel_initializer='he_uniform',
        name="outputs"
    )(x)
    
    model = Model(
        inputs=inputs,
        outputs=outputs,
        name=f"DNN_{hidden_layer}_{data_horizon}"
    )
    model.compile(
        loss=loss,  # Mean Squared Error
        optimizer=optimizer,  # Adam Optimizer
        metrics=metrics  # A variety of error functions
    )

    return model
