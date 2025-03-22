
"""
Model Evaluation Module for Football Prediction
----------------------------------------------
This module handles model evaluation and validation metrics.
"""

import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import KFold

# Import visualization functions
from visualization import confusion_matrix_data, generate_learning_curve

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
    return generate_learning_curve(model, X, y, cv, train_sizes)

def confusion_matrix_metrics(y_true, y_pred, class_names=None):
    """Calculate and format confusion matrix"""
    return confusion_matrix_data(y_true, y_pred, class_names)
