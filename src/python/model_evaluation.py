
"""
Model Evaluation Module for Football Prediction
----------------------------------------------
This module handles model evaluation and validation metrics.
"""

import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import KFold

def evaluate_model(model, X, y, cv=5):
    """Evaluate model using cross-validation"""
    kf = KFold(n_splits=cv, shuffle=True, random_state=42)
    accuracies = []
    precisions = []
    recalls = []
    f1_scores = []
    
    for train_idx, test_idx in kf.split(X):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        
        # Train the model
        model.fit(X_train, y_train)
        
        # Make predictions
        y_pred = model.predict(X_test)
        
        # Calculate metrics
        accuracies.append(accuracy_score(y_test, y_pred))
        precisions.append(precision_score(y_test, y_pred, average='weighted', zero_division=0))
        recalls.append(recall_score(y_test, y_pred, average='weighted', zero_division=0))
        f1_scores.append(f1_score(y_test, y_pred, average='weighted', zero_division=0))
    
    return {
        'accuracy': np.mean(accuracies),
        'precision': np.mean(precisions),
        'recall': np.mean(recalls),
        'f1_score': np.mean(f1_scores),
        'std_accuracy': np.std(accuracies)
    }

def learning_curve(model, X, y, cv=5, train_sizes=None):
    """Generate learning curve data for model"""
    if train_sizes is None:
        # Default: logarithmically spaced train sizes
        train_sizes = np.logspace(
            np.log10(0.1), np.log10(1.0), num=5
        )
    
    n_samples = len(X)
    train_scores = []
    test_scores = []
    
    for train_size in train_sizes:
        actual_train_size = int(train_size * n_samples)
        if actual_train_size < 10:  # Ensure we have at least 10 samples
            actual_train_size = min(10, n_samples)
        
        kf = KFold(n_splits=cv, shuffle=True, random_state=42)
        train_fold_scores = []
        test_fold_scores = []
        
        for train_idx, test_idx in kf.split(X):
            # Use only a subset of training data
            subset_idx = train_idx[:actual_train_size]
            X_train, y_train = X[subset_idx], y[subset_idx]
            X_test, y_test = X[test_idx], y[test_idx]
            
            # Train and evaluate
            model.fit(X_train, y_train)
            train_pred = model.predict(X_train)
            test_pred = model.predict(X_test)
            
            train_fold_scores.append(accuracy_score(y_train, train_pred))
            test_fold_scores.append(accuracy_score(y_test, test_pred))
        
        train_scores.append(np.mean(train_fold_scores))
        test_scores.append(np.mean(test_fold_scores))
    
    return {
        'train_sizes': [int(ts * n_samples) for ts in train_sizes],
        'train_scores': train_scores,
        'test_scores': test_scores
    }

def confusion_matrix_data(y_true, y_pred, class_names=None):
    """Calculate and format confusion matrix"""
    if class_names is None:
        class_names = ["Home Win", "Draw", "Away Win"]
    
    cm = confusion_matrix(y_true, y_pred)
    
    # Calculate row and column sums for percentages
    row_sums = cm.sum(axis=1, keepdims=True)
    
    # Avoid division by zero
    row_sums[row_sums == 0] = 1
    
    # Calculate percentages
    cm_percentage = cm / row_sums * 100
    
    result = {
        'matrix': cm.tolist(),
        'percentage': cm_percentage.tolist(),
        'class_names': class_names
    }
    
    return result
