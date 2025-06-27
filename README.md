Cardio Disease Prediction using Neural Network
This project trains a neural network model to predict cardiovascular disease risk based on a dataset of patient features.

Overview
The dataset (cardio_dataset.csv) contains 7 input features and 1 target variable.

Features and target values are scaled using MinMaxScaler.

The model is a feedforward neural network built with Keras, containing dense layers and dropout for regularization.

The performance metric includes Mean Absolute Error (MAE) and a custom R² score.

The best model is saved during training using model checkpointing.

Final model is saved after training.

Files
cardio_dataset.csv — Input dataset (CSV format).

train_model.py — Script containing data preprocessing, model building, training, and saving.

models/best_model.keras — Best saved model checkpoint during training.

models/final_model.keras — Final saved model after training.

Requirements
Python 3.x

pandas

numpy

scikit-learn

tensorflow (with Keras)

Install dependencies with:

bash
Copy
Edit
pip install pandas numpy scikit-learn tensorflow
Usage
Place cardio_dataset.csv in the same directory as the script.

Run the training script:

bash
Copy
Edit
python train_model.py
The training progress will be displayed along with validation loss.

The best model checkpoint will be saved in models/best_model.keras.

After training, the final model is saved in models/final_model.keras.

Model Architecture
Input layer: 7 features

Dense layer: 128 units, ReLU activation

Dropout: 0.5

Dense layer: 64 units, tanh activation

Dropout: 0.5

Dense layer: 16 units, tanh activation

Output layer: 1 unit, linear activation

Metrics
Loss: Mean Squared Error (MSE)

Metrics: Mean Absolute Error (MAE), Custom R² score

Notes
The R² score is implemented as a custom metric using TensorFlow backend.

The dataset and model paths may need to be adjusted based on your project structure.
