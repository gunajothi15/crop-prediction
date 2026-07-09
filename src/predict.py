import inspect
import joblib
import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt
from pathlib import Path


def predict_crop(n, p, k, temperature, humidity, ph, rainfall):
    """Predict the recommended crop for the given environmental inputs."""
    base_dir = Path(__file__).resolve().parent.parent
    model_path = base_dir / "models" / "random_forest_model.pkl"
    scaler_path = base_dir / "models" / "scaler.pkl"

    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")
    if not scaler_path.exists():
        raise FileNotFoundError(f"Scaler file not found: {scaler_path}")

    clf = joblib.load(model_path)
    scaler = joblib.load(scaler_path)

    feature_names = ["N", "P", "K", "temperature", "humidity", "ph", "rainfall"]
    features = pd.DataFrame(
        [[n, p, k, temperature, humidity, ph, rainfall]],
        columns=feature_names,
        dtype=float,
    )
    scaled_features = scaler.transform(features)
    prediction = clf.predict(scaled_features)[0]

    crop_names = [
        "apple",
        "banana",
        "blackgram",
        "chickpea",
        "coconut",
        "coffee",
        "cotton",
        "grapes",
        "jute",
        "kidneybeans",
        "lentil",
        "maize",
        "mango",
        "mothbeans",
        "mungbean",
        "muskmelon",
        "orange",
        "papaya",
        "pigeonpeas",
        "pomegranate",
        "rice",
        "watermelon",
    ]

    crop = crop_names[int(prediction)] if isinstance(prediction, (int, np.integer)) else prediction
    print(f"Recommended crop to grow: {crop}")

    outputs_dir = base_dir / "outputs"
    outputs_dir.mkdir(exist_ok=True)
    explanation_path = outputs_dir / "single_prediction_explanation.png"

    explainer = shap.TreeExplainer(clf)
    shap_values = explainer.shap_values(scaled_features)

    if hasattr(clf, "classes_") and len(clf.classes_) > 0:
        class_values = np.asarray(clf.classes_)
        if prediction in class_values:
            predicted_class = int(np.where(class_values == prediction)[0][0])
        else:
            predicted_class = int(prediction)
    else:
        predicted_class = int(prediction)

    if isinstance(shap_values, list):
        if predicted_class < len(shap_values):
            values_for_plot = shap_values[predicted_class][0]
            base_value = explainer.expected_value[predicted_class]
        else:
            values_for_plot = shap_values[0][0]
            base_value = explainer.expected_value[0]
    elif isinstance(shap_values, np.ndarray) and shap_values.ndim == 3:
        values_for_plot = shap_values[0, :, predicted_class]
        base_value = explainer.expected_value[predicted_class]
    else:
        values_for_plot = shap_values[0] if isinstance(shap_values, np.ndarray) else shap_values
        base_value = explainer.expected_value

    explanation = shap.Explanation(
        values=values_for_plot,
        base_values=base_value,
        data=scaled_features[0],
        feature_names=feature_names,
    )

    bar_plot = shap.plots.bar
    if "matplotlib" in inspect.signature(bar_plot).parameters:
        bar_plot(explanation, show=False, matplotlib=True)
    else:
        bar_plot(explanation, show=False)
    plt.tight_layout()
    plt.savefig(str(explanation_path), dpi=300, bbox_inches="tight")
    plt.close()
    print(f"SHAP explanation saved to: {explanation_path}")

    return crop


if __name__ == "__main__":
    print("Running sample prediction...")
    
    predict_crop(57,73,85,18.49311205,14.72115044,7.358099622,91.94595352)
