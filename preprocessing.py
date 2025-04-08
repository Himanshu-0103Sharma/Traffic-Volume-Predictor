# Title : Preprocessing the data
# Date : 05-02-2025
# Author : Himanshu Sharma (himanshu.0103sharma@gmail.com)

import numpy as np # For numerical computation
from scipy.io import loadmat # For reading MATLAB dataset
from sklearn.preprocessing import MinMaxScaler # For normalizing data
from sklearn.model_selection import train_test_split # Splitting data

def load_and_preprocess_data(file_path):
    """
    Load and preprocess traffic dataset from a .mat file.

    Args:
        file_path (str): Path to the .mat file.

    Returns:
        tuple: Processed training, validation, and test datasets, along with the scaler for Y.
    """
    # Load the .mat dataset
    data = loadmat(file_path)

    # Extract individual components from the dataset
    tra_X_tr = data["tra_X_tr"][0]  # Training input data (array of sparse matrices, 1261 samples)
    tra_Y_tr = data["tra_Y_tr"]     # Training target data (36 x 1261 matrix)
    tra_X_te = data["tra_X_te"][0]  # Test input data (array of sparse matrices, 840 samples)
    tra_Y_te = data["tra_Y_te"]     # Test target data (36 x 840 matrix)

    # Convert sparse matrices in tra_X_tr and tra_X_te to dense arrays
    # If already dense, no conversion is needed
    tra_X_tr_dense = [x.toarray() if not isinstance(x, np.ndarray) else x for x in tra_X_tr]
    tra_X_te_dense = [x.toarray() if not isinstance(x, np.ndarray) else x for x in tra_X_te]

    # Combine (stack) individual samples into a single 3D array
    # Shape: (samples, timesteps, features) -> (1261, 36, 48) for training data
    X_train_raw = np.stack(tra_X_tr_dense, axis=0)
    X_test_raw = np.stack(tra_X_te_dense, axis=0)

    # Transpose Y data to match the shape of X data: (samples, timesteps)
    # Shape becomes (1261, 36) for training and (840, 36) for test
    Y_train_raw = tra_Y_tr.T
    Y_test_raw = tra_Y_te.T

    # Normalize input (X) and target (Y) data using MinMaxScaler
    num_samples, num_timesteps, feature_size = X_train_raw.shape  # Extract dataset dimensions
    scaler_X = MinMaxScaler()  # Scaler for input data
    scaler_Y = MinMaxScaler()  # Scaler for target data

    # Reshape input data to 2D for normalization (samples * timesteps, features)
    X_train_reshaped = X_train_raw.reshape(-1, feature_size)
    X_test_reshaped = X_test_raw.reshape(-1, feature_size)

    # Fit the scaler on training data and transform both training and test data
    X_train_scaled = scaler_X.fit_transform(X_train_reshaped).reshape(num_samples, num_timesteps, feature_size)
    X_test_scaled = scaler_X.transform(X_test_reshaped).reshape(-1, num_timesteps, feature_size)

    # Normalize target data (Y) as it is already 2D
    Y_train_scaled = scaler_Y.fit_transform(Y_train_raw)
    Y_test_scaled = scaler_Y.transform(Y_test_raw)

    # Split training data into training and validation sets (80% training, 20% validation)
    # Ensures the model has separate data for evaluation during training
    X_train, X_val, Y_train, Y_val = train_test_split(
        X_train_scaled, Y_train_scaled, test_size=0.2, random_state=42
    )

    # Return processed datasets and scaler for Y
    return X_train, X_val, Y_train, Y_val, X_test_scaled, Y_test_scaled, Y_test_raw, scaler_Y



# Example usage
if __name__ == "__main__":
    # Specify the path to the dataset
    file_path = "traffic_dataset.mat"

    # Load and preprocess the dataset
    result = load_and_preprocess_data(file_path)

    # Check if preprocessing was successful
    if result:
        X_train, X_val, Y_train, Y_val, X_test, Y_test, Y_test_raw, scaler_Y = result
        print("Data preprocessing completed successfully.")
