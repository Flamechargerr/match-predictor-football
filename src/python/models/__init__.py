
# Export models
from .base_predictor import FootballPredictor
from .train_utils import train_models
from .prediction import predict_match

# Create a singleton instance for global use
from .base_predictor import predictor
