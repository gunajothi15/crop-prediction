"""
Data Pipeline Module
Handles data loading, preprocessing, and train-test split with proper scaling.
Prevents data leakage by fitting scaler only on training data.
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from pathlib import Path


def get_processed_data():
    """
    Load and process the crop recommendation dataset.
    
    This function:
    - Loads the CSV file
    - Separates features from target
    - Performs 80/20 train-test split
    - Fits StandardScaler only on training data
    - Transforms both training and testing data
    
    Returns:
        tuple: (X_train_scaled, X_test_scaled, y_train, y_test, scaler)
            - X_train_scaled: Scaled training features
            - X_test_scaled: Scaled testing features
            - y_train: Training target labels
            - y_test: Testing target labels
            - scaler: Fitted StandardScaler for future predictions
    """
    
    # Define file path
    data_path = Path(__file__).parent.parent / "Crop_recommendation.csv"
    
    # Load dataset
    df = pd.read_csv(data_path)
    
    # Define the 7 soil/weather features
    feature_columns = ["N", "P", "K", "temperature", "humidity", "ph", "rainfall"]
    target_column = "label"
    
    # Separate features and target
    X = df[feature_columns]
    y = df[target_column]
    
    # 80/20 train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Initialize scaler
    scaler = StandardScaler()
    
    # Fit scaler ONLY on training data to prevent data leakage
    X_train_scaled = scaler.fit_transform(X_train)
    
    # Transform test data using the fitted scaler
    X_test_scaled = scaler.transform(X_test)
    
    return X_train_scaled, X_test_scaled, y_train, y_test, scaler


if __name__ == "__main__":
    # Example usage
    X_train, X_test, y_train, y_test, scaler = get_processed_data()
    print(f"Training set shape: {X_train.shape}")
    print(f"Testing set shape: {X_test.shape}")
    print(f"Scaler parameters: mean={scaler.mean_}, scale={scaler.scale_}")
