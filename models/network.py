from math import floor
from typing import Dict, List
from tensorflow.keras import layers, losses, optimizers, Model, Input, callbacks
from tensorflow.keras.layers import Dropout

from utils.oos_metric import r_squared_fn


def init_model(experiment_name: str, config: dict, input_dim: int) -> Model:
    hidden_layer = config["hidden_layer"]
    num_units = config["num_units"]
    loss = config["loss"]
    optimizer = config["optimizer"]
    model = create_model(
        input_dim=input_dim,
        hidden_layer=hidden_layer,
        num_units=num_units,
        loss=loss,
        optimizer=optimizer,
        experiment_name=experiment_name
    )
    return model


def create_model(
        experiment_name: str,
        input_dim: int = 20,
        hidden_layer: int = 5,
        num_units: int = 40,
        loss: str = "mse",
        optimizer: str = "adam",
        metrics: List[str] = ["mse", "mape", "mae", "msle", r_squared_fn],
) -> Model:

    correction = 50 if input_dim > 100 else 5

    inputs = Input(shape=(input_dim,), name="option_data")
    x = layers.Dense(
        num_units,
        activation='relu',
        kernel_initializer='he_uniform',
        name=f"dense_0"
    )(inputs)
    x = Dropout(0.3)(x)
    print(f"dense_0 layer has {num_units} units")
    for i in range(hidden_layer-1):
        delta = floor(num_units / hidden_layer) * (i+1) + correction
        units = num_units-delta
        x = layers.Dense(
            units,
            activation="relu",
            kernel_initializer='he_uniform',
            name=f"dense_{i+1}"
        )(x)
        x = Dropout(0.1)(x)
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
        name=f"DNN_{hidden_layer}_{experiment_name}"
    )
    model.compile(
        loss=loss,  # Mean Squared Error
        optimizer=optimizer,  # Adam Optimizer
        metrics=metrics  # A variety of error functions
    )

    return model
