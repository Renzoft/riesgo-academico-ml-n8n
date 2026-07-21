"""Definicion canonica de la red neuronal multicapa."""

import tensorflow as tf
from tensorflow.keras.layers import Dense, Dropout, Input
from tensorflow.keras.models import Sequential


def crear_modelo(input_dim: int, num_classes: int) -> Sequential:
    model = Sequential(
        [
            Input(shape=(input_dim,)),
            Dense(64, activation="relu"),
            Dropout(0.20),
            Dense(32, activation="relu"),
            Dropout(0.20),
            Dense(num_classes, activation="softmax"),
        ]
    )
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model
