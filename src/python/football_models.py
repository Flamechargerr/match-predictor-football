
"""
Football Match Prediction Models
--------------------------------
This file contains the Python ML models for football match prediction.
These models are used via Pyodide in the browser.
"""

# Import the prediction and training functions directly
from models.base_predictor import predictor
from models.prediction import predict_match
from models.train_utils import train_and_evaluate_models

# Export the main functions that will be called from JavaScript
__all__ = ['train_models', 'predict_match']

# Define the functions that will be called from JavaScript
def train_models(football_data):
    """Train the models and return performance metrics"""
    try:
        return predictor.train(football_data)
    except Exception as e:
        import traceback
        print(f"Training error: {str(e)}")
        print(traceback.format_exc())
        return []

# We already imported predict_match directly, so no need to redefine it
