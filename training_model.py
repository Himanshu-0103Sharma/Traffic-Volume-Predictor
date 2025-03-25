# Title : Training the LSTM model
# Date : 10-02-2025
# Author : Himanshu Sharma (himanshu.0103sharma@gmail.com)


import tensorflow as tf                                     # for building machine learning model
from tensorflow.keras.models import Sequential              # to create neural network layer by layer
from tensorflow.keras.layers import LSTM, Dense, Dropout    
import preprocessing
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from tensorflow.keras.callbacks import EarlyStopping
import joblib
                               

def create_model(input_shape) :
    
    model = Sequential([
            LSTM(64, activation='relu', input_shape = input_shape, 
                 return_sequences = True), # LSTM layer with 62 memory cell
            Dropout(0.2), # disabling 20% of the data
            LSTM(32, activation='relu', 
                 return_sequences = False), # Another LSTM layer with 32 memory cell
            Dropout(0.2), # disabling 20% of the data
            Dense(36, activation = 'linear')
        ])
    
    return model

def train_model(model, X_train, Y_train, X_val, Y_val, epochs = 100, batch_size = 32):
    early_stopping = EarlyStopping(monitor='val_loss', patience=7, restore_best_weights=True)
    model.compile(optimizer='adam', 
                  loss=tf.keras.losses.MeanSquaredError(), metrics=['mae']) # compiling the model
    history = model.fit(X_train, Y_train, validation_data = (X_val, Y_val), 
                        epochs=epochs, batch_size=batch_size, 
                        callbacks=[early_stopping]) # training the model
    return model,history # returning model and training history

def evaluate_model(model, X_test, Y_test, scaler_Y):
    Y_pred = model.predict(X_test)
    
    # Debug: Print shapes before inverse transform
    print("*************************************************************")
    print("Y_test shape:", Y_test.shape)
    print("Y_pred shape:", Y_pred.shape)
    print("Scaler expected shape:", scaler_Y.n_features_in_)  # Check scaler's fitted shape
    print(isinstance(Y_test, np.ndarray), isinstance(Y_pred, np.ndarray))
    print("*************************************************************")
    
    # Inverse transform prediction to original scale
    Y_pred_original = scaler_Y.inverse_transform(Y_pred)
    Y_test_original = scaler_Y.inverse_transform(Y_test)
    
    # Calculate metrics
    mae = mean_absolute_error(Y_test_original, Y_pred_original)
    mse = mean_squared_error(Y_test_original, Y_pred_original)
    rmse = np.sqrt(mse)
    r2 = r2_score(Y_test_original, Y_pred_original)
    
    print("\nModel Evaluation Metrics:")
    print(f"Mean Absolute Error (MAE): {mae:.4f}")
    print(f"Mean Squared Error (MSE): {mse:.4f}")
    print(f"Root Mean Squared Error (RMSE): {rmse:.4f}")
    print(f"R² Score: {r2:.4f}")

def plot_training_history(history):
    plt.figure(figsize=(12,5)) # 12 width; 5 height
    plt.subplot(1,2,1) # 1 row; 2 colns; 1st plot
    plt.plot(history.history['loss'], label='Training Loss') # error on training the data
    plt.plot(history.history['val_loss'], label='Validation Loss') # error on validating the data
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Model Loss Over Epochs')
    
    plt.subplot(1,2,2) # 1 row; 2 colns; 2nd plot
    plt.plot(history.history['mae'], label='Training MAE') # MAE : How prediction is far from actual data
    plt.plot(history.history['val_mae'], label='Validation MAE')
    plt.xlabel('Epochs')
    plt.ylabel('MAE')
    plt.legend()
    plt.title('Model MAE over Epochs')
    
    plt.show()
    
    
if __name__ == "__main__":
    file_path = "traffic_dataset.mat"
    X_train, X_val, Y_train, Y_val, X_test, Y_test, Y_test_raw, scaler_Y = preprocessing.load_and_preprocess_data(file_path)
    
    input_shape = (X_train.shape[1], X_train.shape[2]) # Dynamically get input shape
    model = create_model(input_shape)
    model, history = train_model(model, X_train, Y_train, X_val, Y_val)
    
    #Save the model and scaler_Y
    model.save("traffic_volume_lstm_model.h5")
    joblib.dump(scaler_Y, "scaler_Y.pkl")
    
    #Evalute Model
    evaluate_model(model, X_test, Y_test, scaler_Y)
    
    #Plot training history
    plot_training_history(history)
    
    
        


    
    