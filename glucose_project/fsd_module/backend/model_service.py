import numpy as np
import tensorflow as tf
from tensorflow import keras
import os

# --- Dummy Model Path (Replace with your actual trained model) ---
# NOTE: Ensure this path is correct relative to the main.py execution location.
MODEL_PATH = 'prediction_model.h5' 

# --- Simplified Feature Preparation (Simulating the DA logic) ---
def prepare_features(current_bg, time_since_meal, current_activity):
    """
    Simulates the feature preparation step.
    In a real scenario, this would create a sequence of features (12xN)
    from the raw historical data.
    """
    # Create a dummy feature vector for demonstration
    # [Current BG, Time since meal, Activity status, Simplified Lagged BG values...]
    features = np.array([
        current_bg,
        time_since_meal,
        1.0 if current_activity else 0.0,
        current_bg * 0.9,
        current_bg * 0.95,
        current_bg * 1.05
    ]).astype(np.float32)

    # Reshape for LSTM input: (Batch Size, Timesteps, Features)
    return features.reshape(1, 1, -1)


class PredictionService:
    def __init__(self):
        self.model = self._load_model()
        print("Prediction Service Initialized.")

    def _load_model(self):
        """Load the trained LSTM model or use a dummy for demonstration."""
        if os.path.exists(MODEL_PATH):
            # In your M.Tech project, you will use:
            # return keras.models.load_model(MODEL_PATH)
            print(f"Loading real model from {MODEL_PATH}.")
        else:
            print(f"Model file not found at {MODEL_PATH}. Using DUMMY Model.")
        
        # --- DUMMY MODEL IMPLEMENTATION ---
        class DummyModel:
            def predict(self, features):
                # Simple simulation: BG trend is 5% increase + small noise
                current_bg = features[0, 0, 0]
                predicted_bg = current_bg * 1.05 + np.random.uniform(-5, 5)
                return np.array([[predicted_bg]])
        return DummyModel()

    def get_prediction(self, current_bg: float, time_since_meal: float, current_activity: bool):
        """
        Main function to predict BG_t+30 and assess risk.
        """
        features = prepare_features(current_bg, time_since_meal, current_activity)
        
        # Predict using the (dummy/real) model
        predicted_bg = self.model.predict(features)[0][0]

        # Risk Assessment Logic: Predict a high spike if the BG is rising sharply towards high levels.
        if predicted_bg > 180 and current_bg < 150:
            risk_level = "HIGH"
            risk_message = "⚠️ Significant spike predicted in 30 mins."
        elif predicted_bg > 150:
            risk_level = "MEDIUM"
            risk_message = "🔺 Elevated BG predicted. Be aware."
        else:
            risk_level = "LOW"
            risk_message = "✅ BG predicted to stay within range."

        return predicted_bg, risk_level, risk_message

# Global instance of the service
prediction_service = PredictionService()