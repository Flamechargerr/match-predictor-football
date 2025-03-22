
import { MatchPrediction, ModelPerformance } from '@/types';
import { toast } from "@/components/ui/use-toast";

// Type definition for the Pyodide module
declare global {
  interface Window {
    loadPyodide: (options: { indexURL: string }) => Promise<any>;
  }
}

class PyodideService {
  private pyodide: any = null;
  private isInitialized = false;
  private isInitializing = false;
  private modelPerformance: ModelPerformance[] = [];
  private pythonCode = `
import numpy as np
import sys
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score

# Feature names
feature_names = [
    'home_goals', 'away_goals', 'home_shots', 'away_shots', 
    'home_shots_target', 'away_shots_target', 'home_red_cards', 'away_red_cards'
]

# Model wrappers
class NaiveBayesModel:
    def __init__(self):
        self.model = GaussianNB()
        self.trained = False
        
    def train(self, X, y):
        self.model.fit(X, y)
        self.trained = True
        
    def predict(self, X):
        if not self.trained:
            return {"prediction": 0, "probs": [0.33, 0.33, 0.33]}
        probs = self.model.predict_proba(X)[0]
        prediction = np.argmax(probs)
        return {"prediction": int(prediction), "probs": probs.tolist()}
    
    def evaluate(self, X, y):
        preds = self.model.predict(X)
        accuracy = accuracy_score(y, preds)
        precision = precision_score(y, preds, average=None, zero_division=0)
        return {"accuracy": accuracy, "precision": precision}

class RandomForestModel:
    def __init__(self):
        self.model = RandomForestClassifier(n_estimators=10, max_depth=5, random_state=42)
        self.trained = False
        
    def train(self, X, y):
        self.model.fit(X, y)
        self.trained = True
        
    def predict(self, X):
        if not self.trained:
            return {"prediction": 0, "probs": [0.33, 0.33, 0.33]}
        probs = self.model.predict_proba(X)[0]
        prediction = np.argmax(probs)
        return {"prediction": int(prediction), "probs": probs.tolist()}
    
    def evaluate(self, X, y):
        preds = self.model.predict(X)
        accuracy = accuracy_score(y, preds)
        precision = precision_score(y, preds, average=None, zero_division=0)
        return {"accuracy": accuracy, "precision": precision}

class LogisticRegressionModel:
    def __init__(self):
        self.model = LogisticRegression(max_iter=1000, random_state=42)
        self.trained = False
        
    def train(self, X, y):
        self.model.fit(X, y)
        self.trained = True
        
    def predict(self, X):
        if not self.trained:
            return {"prediction": 0, "probs": [0.33, 0.33, 0.33]}
        probs = self.model.predict_proba(X)[0]
        prediction = np.argmax(probs)
        return {"prediction": int(prediction), "probs": probs.tolist()}
    
    def evaluate(self, X, y):
        preds = self.model.predict(X)
        accuracy = accuracy_score(y, preds)
        precision = precision_score(y, preds, average=None, zero_division=0)
        return {"accuracy": accuracy, "precision": precision}

# Create model instances
naive_bayes = NaiveBayesModel()
random_forest = RandomForestModel()
logistic_regression = LogisticRegressionModel()

# Main functions to be called from JavaScript
def train_models(football_data):
    # Split data into features and target
    data = np.array(football_data)
    X = data[:, :8]  # First 8 columns are features
    y = data[:, 8]   # Last column is the target
    
    # Split into train/test (80/20)
    np.random.seed(42)
    indices = np.random.permutation(len(X))
    train_size = int(0.8 * len(X))
    train_indices = indices[:train_size]
    test_indices = indices[train_size:]
    
    X_train, y_train = X[train_indices], y[train_indices]
    X_test, y_test = X[test_indices], y[test_indices]
    
    # Train models
    naive_bayes.train(X_train, y_train)
    random_forest.train(X_train, y_train)
    logistic_regression.train(X_train, y_train)
    
    # Evaluate
    nb_metrics = naive_bayes.evaluate(X_test, y_test)
    rf_metrics = random_forest.evaluate(X_test, y_test)
    lr_metrics = logistic_regression.evaluate(X_test, y_test)
    
    # Calculate average precision
    nb_precision = np.mean(nb_metrics["precision"])
    rf_precision = np.mean(rf_metrics["precision"])
    lr_precision = np.mean(lr_metrics["precision"])
    
    return [
        {"name": "Naive Bayes", "accuracy": float(nb_metrics["accuracy"]), "precision": float(nb_precision)},
        {"name": "Random Forest", "accuracy": float(rf_metrics["accuracy"]), "precision": float(rf_precision)},
        {"name": "Logistic Regression", "accuracy": float(lr_metrics["accuracy"]), "precision": float(lr_precision)}
    ]

def predict_match(match_data):
    X = np.array([match_data])
    
    # Get predictions
    nb_result = naive_bayes.predict(X)
    rf_result = random_forest.predict(X)
    lr_result = logistic_regression.predict(X)
    
    # Convert predictions to outcomes
    outcomes = ["Home Win", "Draw", "Away Win"]
    
    return [
        {
            "modelName": "Naive Bayes",
            "outcome": outcomes[nb_result["prediction"]],
            "confidence": float(np.max(nb_result["probs"]) * 100),
            "probabilities": nb_result["probs"]
        },
        {
            "modelName": "Random Forest",
            "outcome": outcomes[rf_result["prediction"]],
            "confidence": float(np.max(rf_result["probs"]) * 100),
            "probabilities": rf_result["probs"]
        },
        {
            "modelName": "Logistic Regression",
            "outcome": outcomes[lr_result["prediction"]],
            "confidence": float(np.max(lr_result["probs"]) * 100),
            "probabilities": lr_result["probs"]
        }
    ]
`;

  constructor() {
    this.initializePyodide();
  }

  private async initializePyodide(): Promise<void> {
    if (this.isInitialized || this.isInitializing) {
      return;
    }

    this.isInitializing = true;

    try {
      // Load Pyodide script
      if (!window.loadPyodide) {
        const script = document.createElement('script');
        script.src = 'https://cdn.jsdelivr.net/pyodide/v0.23.4/full/pyodide.js';
        document.head.appendChild(script);

        // Wait for the script to load
        await new Promise<void>((resolve) => {
          script.onload = () => resolve();
        });
      }

      // Load Pyodide
      console.log('Loading Pyodide...');
      this.pyodide = await window.loadPyodide({
        indexURL: 'https://cdn.jsdelivr.net/pyodide/v0.23.4/full/',
      });

      // Install scikit-learn package
      console.log('Installing scikit-learn...');
      await this.pyodide.loadPackage(['scikit-learn', 'numpy']);

      // Run the Python code
      await this.pyodide.runPythonAsync(this.pythonCode);

      this.isInitialized = true;
      this.isInitializing = false;
      console.log('Pyodide initialized successfully');
    } catch (error) {
      console.error('Error initializing Pyodide:', error);
      this.isInitializing = false;
      toast({
        title: "Error",
        description: "Failed to initialize Python environment. Please refresh the page and try again.",
        variant: "destructive",
      });
    }
  }

  public async trainModels(footballData: number[][]): Promise<ModelPerformance[]> {
    await this.ensureInitialized();

    try {
      console.log('Training models with Python...');
      // Convert data to Python
      this.pyodide.globals.set('football_data', footballData);
      
      // Train models and get performance metrics
      const result = await this.pyodide.runPythonAsync(`
        import json
        results = train_models(football_data)
        json.dumps(results)
      `);
      
      this.modelPerformance = JSON.parse(result);
      console.log('Model performance:', this.modelPerformance);
      
      return this.modelPerformance;
    } catch (error) {
      console.error('Error training models:', error);
      toast({
        title: "Training Error",
        description: "An error occurred while training the models. Please try again.",
        variant: "destructive",
      });
      return [];
    }
  }

  public async predictMatch(matchData: number[]): Promise<MatchPrediction[]> {
    await this.ensureInitialized();

    try {
      console.log('Predicting with Python models...');
      // Convert data to Python
      this.pyodide.globals.set('match_data', matchData);
      
      // Get predictions
      const result = await this.pyodide.runPythonAsync(`
        import json
        predictions = predict_match(match_data)
        json.dumps(predictions)
      `);
      
      const predictions: MatchPrediction[] = JSON.parse(result);
      
      // Add model accuracy from our stored performance metrics
      return predictions.map(prediction => {
        const modelPerf = this.modelPerformance.find(p => p.name === prediction.modelName);
        return {
          ...prediction,
          modelAccuracy: modelPerf ? modelPerf.accuracy * 100 : 75 // default to 75% if not found
        };
      });
    } catch (error) {
      console.error('Error predicting match:', error);
      toast({
        title: "Prediction Error",
        description: "An error occurred while making predictions. Please try again.",
        variant: "destructive",
      });
      return [];
    }
  }

  public getModelPerformance(): ModelPerformance[] {
    return this.modelPerformance;
  }

  private async ensureInitialized(): Promise<void> {
    if (!this.isInitialized) {
      if (!this.isInitializing) {
        await this.initializePyodide();
      } else {
        // Wait for initialization to complete
        await new Promise<void>((resolve) => {
          const checkInterval = setInterval(() => {
            if (this.isInitialized) {
              clearInterval(checkInterval);
              resolve();
            }
          }, 100);
        });
      }
    }
  }
}

// Export as singleton
export const pyodideService = new PyodideService();
