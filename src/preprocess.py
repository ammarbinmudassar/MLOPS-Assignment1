import os

import numpy as np
import yaml
from sklearn.model_selection import train_test_split

params = yaml.safe_load(open("params.yaml"))["preprocess"]


def normalize(x):
    return x.astype("float16") / 255.0


train = np.load("data/raw/train.npz")
test = np.load("data/raw/test.npz")

x_train, x_val, y_train, y_val = train_test_split(
    normalize(train["x"]),
    train["y"],
    test_size=params["test_size"],
    random_state=params["seed"],
    stratify=train["y"],
)
x_test, y_test = normalize(test["x"]), test["y"]

os.makedirs("data/processed", exist_ok=True)
np.savez("data/processed/train.npz", x=x_train, y=y_train)
np.savez("data/processed/val.npz", x=x_val, y=y_val)
np.savez("data/processed/test.npz", x=x_test, y=y_test)

print(f"Processed: train={x_train.shape}, val={x_val.shape}, test={x_test.shape}")
