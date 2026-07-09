"""
Explainability Module
Uses SHAP to explain the Random Forest model predictions on crop classification.
Generates feature importance visualizations for interpretability.
"""

from src.data_pipeline import get_processed_data
from src.train import train_model
import shap
import matplotlib.pyplot as plt
import joblib
from pathlib import Path


def generate_shap_analysis():
    """
    Generate SHAP analysis and save summary plot.
    """
    
    # Check if model is already trained; if not, train it
    models_dir = Path(__file__).parent.parent / "models"
    model_path = models_dir / "random_forest_model.pkl"
    
    if model_path.exists():
        print("Loading pre-trained model...")
        clf = joblib.load(model_path)
    else:
        print("Training model...")
        clf, _ = train_model()
    
    # Load processed data
    print("Loading and processing data...")
    X_train_scaled, X_test_scaled, y_train, y_test, scaler = get_processed_data()
    
    # Create SHAP TreeExplainer
    print("Creating SHAP TreeExplainer...")
    explainer = shap.TreeExplainer(clf)
    
    # Calculate SHAP values for test set
    print("Calculating SHAP values for test set...")
    shap_values = explainer.shap_values(X_test_scaled)
    
    # Get feature names
    feature_names = ["N", "P", "K", "temperature", "humidity", "ph", "rainfall"]
    
    # Create outputs directory
    outputs_dir = Path(__file__).parent.parent / "outputs"
    outputs_dir.mkdir(exist_ok=True)
    
    # Debug: Check shape of shap_values
    print(f"Debug - shap_values type: {type(shap_values)}")
    if isinstance(shap_values, list):
        print(f"Debug - shap_values is a list with {len(shap_values)} classes")
        print(f"Debug - Each class shape: {shap_values[0].shape}")
    else:
        print(f"Debug - shap_values shape: {shap_values.shape}")
    
    # Generate and save SHAP summary plot
    print("Generating SHAP summary plot...")
    
    # shap_values shape: (n_samples, n_features, n_classes)
    # We need to compute feature importance across all samples and classes
    # Mean absolute SHAP value per feature (averaged over samples and classes)
    feature_importance = abs(shap_values).mean(axis=0).mean(axis=1)
    
    print(f"Debug - feature_importance shape after aggregation: {feature_importance.shape}")
    print(f"Debug - n_features: {len(feature_names)}")
    
    # Sort features by importance
    indices = feature_importance.argsort()[::-1]
    
    # Create summary plot
    fig, ax = plt.subplots(figsize=(10, 6))
    
    ax.barh(range(len(indices)), feature_importance[indices])
    ax.set_yticks(range(len(indices)))
    ax.set_yticklabels([feature_names[i] for i in indices])
    ax.set_xlabel("Mean |SHAP value|")
    ax.set_title("SHAP Feature Importance for Crop Prediction Model")
    ax.grid(axis='x', alpha=0.3)
    
    plot_path = outputs_dir / "shap_summary.png"
    plt.tight_layout()
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"✓ SHAP summary plot saved to {plot_path}")
    plt.close()
    
    # Also create a detailed beeswarm plot for better visualization
    print("Generating detailed SHAP beeswarm plot...")
    import matplotlib.pyplot as plt
    plt.figure(figsize=(10, 6))

    # Pass show=False to prevent SHAP from clearing the canvas
    shap.plots.beeswarm(shap_values[:, :, 0], show=False)

    beeswarm_path = outputs_dir / "shap_beeswarm_detailed.png"
    plt.savefig(beeswarm_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ SHAP beeswarm plot saved to {beeswarm_path}")
    
    print("\n" + "="*60)
    print("SHAP ANALYSIS COMPLETE")
    print("="*60)
    print(f"Outputs saved to: {outputs_dir}")
    print(f"  - shap_summary.png: Feature importance bar plot")
    print(f"  - shap_beeswarm_detailed.png: Detailed beeswarm plot")
    print("="*60 + "\n")


if __name__ == "__main__":
    generate_shap_analysis()
