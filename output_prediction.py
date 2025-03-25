# Title : Output prediction; Comparing the model output with the actual output
# Date : 22/02/2025
# Author : Himanshu Sharm (himanshu.0103sharma@gmail.com)

import numpy as np # for numerical operations
import tensorflow as tf # machine learning framework
import preprocessing
import matplotlib.pyplot as plt
import pandas as pd
import joblib

def load_model(model_path):
    return tf.keras.models.load_model(model_path)

def load_scaler(scaler_path):
    return joblib.load(scaler_path)

def predict_traffic(model, X_test, scaler_Y):
    predictions = model.predict(X_test)
    predictions_rescaled = scaler_Y.inverse_transform(predictions) # Rescale to original
    return predictions_rescaled

# Visualize predictions vs actual values
def plot_predictions(predictions, actual_values, num_samples=10):
    plt.figure(figsize=(12, 6))
    for i in range(num_samples):  # Plot only a subset of samples for clarity
        plt.plot(actual_values[i], label=f"Actual {i+1}", linestyle='dashed', alpha=0.7)
        plt.plot(predictions[i], label=f"Predicted {i+1}", alpha=0.9)
    
    plt.xlabel("Time Step")
    plt.ylabel("Traffic Volume")
    plt.title("Predicted vs Actual Traffic Volume")
    plt.legend(loc='upper right', fontsize='small', ncol=2, frameon=True)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.show()
    
def save_prediction_csv(predictions, actual_values, filename = "predictions.csv"):
    df = pd.DataFrame({"Predicted": predictions.flatten(), "Actual": actual_values.flatten()})
    df.to_csv(filename, index=False)
    print(f"Predictions saved to {filename}")
    
if __name__ == "__main__":
    file_path = "traffic_dataset.mat"
    model_path = "traffic_volume_lstm_model.h5"
    scaler_path = "scaler_Y.pkl"
    
    # Preprocess the data and load the model
    _, _, _, _, X_test, Y_test_scaled, Y_test_raw, _ = preprocessing.load_and_preprocess_data(file_path)
    model = load_model(model_path)
    scaler_Y = load_scaler(scaler_path)
    
    predictions = predict_traffic(model, X_test, scaler_Y)
    
    # Compare a few predictions with actual values
    for i in range(5):  # Display the first 5 predictions
        print(f"Predicted: {predictions[i]}, Actual: {Y_test_raw[i]}")
        
    plot_predictions(predictions, Y_test_raw)
    
    save_prediction_csv(predictions, Y_test_raw)



