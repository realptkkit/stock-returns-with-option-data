from typing import Dict, List
import tensorflow as tf
from tensorflow.keras import layers, losses, optimizers, Model, Input
from keras.layers import Dropout


def get_params(config: Dict()):
    pass


def create_model(
        input_dim: int = 938,
        num_layers: int = 5,
        loss: str = "mse",
        optimizer: str = "adam",
        metrics: List(str) = ["mse", "mapa", "mae", "msle"]
) -> Model:
    inputs = Input(shape=(input_dim,), name="Option Data")
    x = layers.Dense(200, activation='relu',
                     kernel_initializer='he_uniform')(inputs)
    x = Dropout(0.4)(x)
    for i in num_layers:
        x = layers.Dense(200, activation="relu",
                         kernel_initializer='he_uniform', name=f"dense_{i}")(x)
        x = Dropout(0.3)(x)
    outputs = layers.Dense(1, activation="linear",
                           kernel_initializer='he_uniform', name="outputs")

    model = Model(inputs=inputs, outputs=outputs,
                  name=f"DNN_{num_layers}")
    model.compile(
        loss=loss,  # Mean Squared Error
        optimizer=optimizer,  # Adam Optimizer
        metrics=metrics  # A variety of error functions
    )

    return model
