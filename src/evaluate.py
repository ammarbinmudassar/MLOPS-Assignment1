import json

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn.metrics import ConfusionMatrixDisplay

CLASSES = ["T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
           "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"]

model = tf.keras.models.load_model("models/model.h5")
test = np.load("data/processed/test.npz")

loss, accuracy = model.evaluate(test["x"], test["y"], verbose=0)
predictions = model.predict(test["x"], verbose=0).argmax(axis=1)

ConfusionMatrixDisplay.from_predictions(test["y"], predictions, display_labels=CLASSES, xticks_rotation=45)
plt.tight_layout()
plt.savefig("confusion_matrix.png")

metrics = {"test_loss": round(float(loss), 4), "test_accuracy": round(float(accuracy), 4)}
with open("metrics.json", "w") as f:
    json.dump(metrics, f, indent=2)

print(metrics)
