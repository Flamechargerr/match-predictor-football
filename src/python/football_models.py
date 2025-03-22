
"""
Football Match Prediction Models
--------------------------------
This file contains the Python ML models for football match prediction.
These models are used via Pyodide in the browser.
"""

# Import from our new model modules
from models import train_models, predict_match, predictor

# Export the main functions that will be called from JavaScript
__all__ = ['train_models', 'predict_match']
