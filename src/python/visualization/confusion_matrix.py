
"""
Confusion Matrix Visualization
-----------------------------
Functions to generate and visualize confusion matrices.
"""

import numpy as np
from sklearn.metrics import confusion_matrix

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

def plot_confusion_matrix(y_true, y_pred, class_names=None, figsize=(8, 6)):
    """Plot confusion matrix visualization for model evaluation"""
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns
        
        if class_names is None:
            class_names = ["Home Win", "Draw", "Away Win"]
        
        cm = confusion_matrix(y_true, y_pred)
        
        plt.figure(figsize=figsize)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=class_names, yticklabels=class_names)
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        plt.title('Confusion Matrix')
        
        return plt
    except ImportError:
        print("Matplotlib and/or Seaborn not available")
        return None
