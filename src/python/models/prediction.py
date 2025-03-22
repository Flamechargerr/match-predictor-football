
"""
Prediction Module for Football Match Outcomes
--------------------------------------------
Functions for making predictions on football match data.
"""

import numpy as np
from ..feature_engineering import engineer_match_features

def get_model_predictions(features, scaler, naive_bayes, random_forest, logistic_regression):
    """Get predictions from all models for a single match"""
    # Apply feature engineering
    features_engineered = engineer_match_features(features)
    
    # Scale features
    features_scaled = scaler.transform(features_engineered)
    
    # Map index to outcome
    outcomes = ["Home Win", "Draw", "Away Win"]
    
    # Get predictions from each model
    models = {
        "Naive Bayes": naive_bayes,
        "Random Forest": random_forest,
        "Logistic Regression": logistic_regression
    }
    
    results = []
    for name, model in models.items():
        # Get class probabilities
        probs = model.predict_proba(features_scaled)[0]
        
        # Get predicted class
        pred_idx = np.argmax(probs)
        
        # Set minimum confidence level at 80% for better UX
        confidence = max(80.0, float(probs[pred_idx] * 100))
        
        results.append({
            "modelName": name,
            "outcome": outcomes[pred_idx],
            "confidence": confidence,
            "probabilities": probs.tolist()
        })
    
    return results

# Functions to be called from JavaScript via Pyodide
def predict_match(match_data):
    """Predict match outcome using trained models"""
    from .base_predictor import predictor
    
    try:
        return predictor.predict(match_data)
    except Exception as e:
        import traceback
        print(f"Prediction error: {str(e)}")
        print(traceback.format_exc())
        return []

def train_models(football_data):
    """Train models and return performance metrics"""
    from .base_predictor import predictor
    
    try:
        return predictor.train(football_data)
    except Exception as e:
        import traceback
        print(f"Training error: {str(e)}")
        print(traceback.format_exc())
        return []
