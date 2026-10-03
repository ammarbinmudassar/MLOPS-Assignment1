import os

import numpy as np
import tensorflow as tf
import yaml

params = yaml.safe_load(open("params.yaml"))["train"]
tf.keras.utils.set_random_seed(params["seed"])

train = np.load("data/processed/train.npz")
val = np.load("data/processed/val.npz")

model = tf.keras.Sequential([
    tf.keras.Input(shape=(28, 28)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(params["dense_units"], activation="relu"),
    tf.keras.layers.Dropout(params["dropout_rate"]),
    tf.keras.layers.Dense(10, activation="softmax"),
])
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=params["learning_rate"]),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
)

os.makedirs("models", exist_ok=True)
model.fit(
    train["x"], train["y"],
    validation_data=(val["x"], val["y"]),
    epochs=params["epochs"],
    batch_size=params["batch_size"],
    callbacks=[tf.keras.callbacks.CSVLogger("models/history.csv")],
)
model.save("models/model.h5")
print("Saved model to models/model.h5")
# trains the ANN using values from params.yaml
