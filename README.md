# fashion-ann-pipeline

Classifies Fashion-MNIST images with a simple fully-connected ANN (TensorFlow/Keras).
Data and models are versioned with DVC (Google Drive remote) and the whole
pipeline runs with a single `dvc repro`.

## Setup

```bash
python3.11 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Run

```bash
dvc pull      # get data/models from Google Drive
dvc repro     # prepare -> preprocess -> train -> evaluate
dvc metrics show
```
