"""
Training Module
Trains a RandomForestClassifier with cross-validation and evaluation.
"""

from src.data_pipeline import get_processed_data
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import classification_report
import joblib
from pathlib import Path
import numpy as np


def train_model():
    """
    Load data, perform cross-validation, train final model, and evaluate.
    """
    
    # Load and preprocess data
    print("Loading and processing data...")
    X_train_scaled, X_test_scaled, y_train, y_test, scaler = get_processed_data()
    print(f"Training set shape: {X_train_scaled.shape}")
    print(f"Testing set shape: {X_test_scaled.shape}\n")
    
    # Initialize RandomForestClassifier
    clf = RandomForestClassifier(random_state=42, n_jobs=-1)
    
    # 5-Fold Stratified Cross-Validation
    print("Running 5-Fold Stratified Cross-Validation...")
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = cross_val_score(clf, X_train_scaled, y_train, cv=skf, scoring='accuracy')
    
    print(f"Cross-Validation Scores: {cv_scores}")
    print(f"Mean CV Accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})\n")
    
    # Train final model on full training set
    print("Training final model on full training set...")
    clf.fit(X_train_scaled, y_train)
    
    # Predict on test set
    y_pred = clf.predict(X_test_scaled)
    
    # Print comprehensive classification report
    print("\n" + "="*60)
    print("CLASSIFICATION REPORT (Test Set)")
    print("="*60)
    print(classification_report(y_test, y_pred, digits=4))
    
    # Print macro-averaged F1-score explicitly
    report_dict = classification_report(y_test, y_pred, output_dict=True)
    macro_f1 = report_dict['macro avg']['f1-score']
    print(f"Macro-Averaged F1-Score: {macro_f1:.4f}")
    print("="*60 + "\n")
    
    # Save model and scaler
    models_dir = Path(__file__).parent.parent / "models"
    models_dir.mkdir(exist_ok=True)
    
    joblib.dump(clf, models_dir / "random_forest_model.pkl")
    joblib.dump(scaler, models_dir / "scaler.pkl")
    print(f"Model saved to {models_dir / 'random_forest_model.pkl'}")
    print(f"Scaler saved to {models_dir / 'scaler.pkl'}\n")
    
    return clf, scaler


if __name__ == "__main__":
    train_model()
