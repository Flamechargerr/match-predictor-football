
"""
Advanced Feature Engineering for Football Match Prediction
---------------------------------------------------------
This module handles feature engineering for the football prediction models.
"""

import numpy as np

def engineer_match_features(X):
    """Create sophisticated derived features from raw match statistics"""
    n_samples = X.shape[0]
    features = np.zeros((n_samples, 20))  # 8 original + 12 derived features
    
    # Copy original features
    features[:, :8] = X
    
    # Extract components for readability
    home_goals = X[:, 0]
    away_goals = X[:, 1]
    home_shots = X[:, 2]
    away_shots = X[:, 3]
    home_shots_target = X[:, 4]
    away_shots_target = X[:, 5]
    home_red_cards = X[:, 6]
    away_red_cards = X[:, 7]
    
    # Basic derived features
    features[:, 8] = home_goals - away_goals  # Goal difference
    features[:, 9] = home_shots - away_shots  # Shot difference
    features[:, 10] = home_shots_target - away_shots_target  # Shots on target difference
    features[:, 11] = home_red_cards - away_red_cards  # Red card difference
    
    # Shot efficiency (shots on target / total shots)
    shot_eff_home = np.divide(home_shots_target, home_shots, 
                             out=np.zeros_like(home_shots_target), 
                             where=home_shots!=0)
    shot_eff_away = np.divide(away_shots_target, away_shots, 
                             out=np.zeros_like(away_shots_target), 
                             where=away_shots!=0)
    features[:, 12] = shot_eff_home
    features[:, 13] = shot_eff_away
    features[:, 14] = shot_eff_home - shot_eff_away  # Efficiency difference
    
    # Scoring efficiency (goals / shots on target)
    score_eff_home = np.divide(home_goals, home_shots_target, 
                              out=np.zeros_like(home_goals), 
                              where=home_shots_target!=0)
    score_eff_away = np.divide(away_goals, away_shots_target, 
                              out=np.zeros_like(away_goals), 
                              where=away_shots_target!=0)
    features[:, 15] = score_eff_home
    features[:, 16] = score_eff_away
    features[:, 17] = score_eff_home - score_eff_away  # Scoring efficiency difference
    
    # Advanced features
    
    # Red card impact (goals per shot with red card penalty)
    rc_penalty_home = 1.0 / (1.0 + home_red_cards * 0.5)  # Diminishing returns for multiple red cards
    rc_penalty_away = 1.0 / (1.0 + away_red_cards * 0.5)
    
    features[:, 18] = np.divide(home_goals, home_shots, 
                               out=np.zeros_like(home_goals), 
                               where=home_shots!=0) * rc_penalty_home
    
    features[:, 19] = np.divide(away_goals, away_shots, 
                               out=np.zeros_like(away_goals), 
                               where=away_shots!=0) * rc_penalty_away
    
    return features

def normalize_features(X, with_mean=True, with_std=True):
    """Normalize features to have zero mean and unit variance"""
    if with_mean:
        feature_means = np.mean(X, axis=0)
        X = X - feature_means
    
    if with_std:
        feature_stds = np.std(X, axis=0)
        # Avoid division by zero
        feature_stds[feature_stds < 1e-10] = 1.0
        X = X / feature_stds
    
    return X

def add_polynomial_features(X, degree=2, interaction_only=True):
    """Add polynomial features up to the specified degree"""
    n_samples, n_features = X.shape
    
    if interaction_only:
        # Only add interaction terms (products of features)
        poly_features = []
        for i in range(n_features):
            for j in range(i+1, n_features):
                poly_features.append(X[:, i] * X[:, j])
        
        if len(poly_features) > 0:
            poly_features = np.column_stack(poly_features)
            return np.hstack((X, poly_features))
        return X
    else:
        # Full polynomial expansion
        from sklearn.preprocessing import PolynomialFeatures
        poly = PolynomialFeatures(degree=degree, include_bias=False)
        return poly.fit_transform(X)

def augment_training_data(X, y, noise_level=0.05, n_samples=1):
    """Create synthetic data augmentations with small noise"""
    X_augmented = [X]
    y_augmented = [y]
    
    for _ in range(n_samples):
        # Add small random noise to features
        noise = np.random.normal(0, noise_level, X.shape) * X
        X_noisy = X + noise
        
        X_augmented.append(X_noisy)
        y_augmented.append(y)
    
    return np.vstack(X_augmented), np.hstack(y_augmented)
