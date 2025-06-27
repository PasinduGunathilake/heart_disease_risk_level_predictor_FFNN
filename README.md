# Heart Disease Risk Level Prediction Neural Network

This project trains a neural network to predict cardiovascular disease risk based on patient features from the `cardio_dataset.csv`.

---

## Overview

- Uses a dataset with 7 features and 1 target variable.
- Applies MinMax scaling to inputs and targets.
- Neural network built with Keras featuring Dense and Dropout layers.
- Tracks Mean Absolute Error (MAE) and a custom R² score metric.
- Saves the best model during training with a checkpoint callback.
- Final trained model saved after completion.

---

## Files

- `cardio_dataset.csv` — Input dataset in CSV format.
- `train_model.py` — Python script for data processing, model training, and saving.
- `models/best_model.keras` — Checkpointed best model during training.
- `models/final_model.keras` — Final saved model after training.

---

## Installation

Make sure you have Python 3.x installed. Then install required packages:

```bash
pip install pandas numpy scikit-learn tensorflow
