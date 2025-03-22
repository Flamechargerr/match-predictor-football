
"""
Learning Curve Visualization
---------------------------
Functions to generate and visualize learning curves for models.
"""

import numpy as np
from sklearn.model_selection import KFold

def generate_learning_curve(model, X, y, cv=5, train_sizes=None):
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
            
            from sklearn.metrics import accuracy_score
            train_fold_scores.append(accuracy_score(y_train, train_pred))
            test_fold_scores.append(accuracy_score(y_test, test_pred))
        
        train_scores.append(np.mean(train_fold_scores))
        test_scores.append(np.mean(test_fold_scores))
    
    return {
        'train_sizes': [int(ts * n_samples) for ts in train_sizes],
        'train_scores': train_scores,
        'test_scores': test_scores
    }

def plot_learning_curve(model, X, y, cv=5, train_sizes=None, figsize=(10, 6)):
    """Plot learning curve to visualize model performance with increasing data"""
    try:
        import matplotlib.pyplot as plt
        
        result = generate_learning_curve(model, X, y, cv, train_sizes)
        
        plt.figure(figsize=figsize)
        plt.plot(result['train_sizes'], result['train_scores'], 'o-', label='Training score')
        plt.plot(result['train_sizes'], result['test_scores'], 'o-', label='Test score')
        plt.title('Learning Curve')
        plt.xlabel('Training examples')
        plt.ylabel('Accuracy')
        plt.grid(True)
        plt.legend(loc='best')
        
        return plt
    except ImportError:
        print("Matplotlib not available")
        return None
