
import { MLModel, calculateMetrics } from './ModelInterface';

// Naive Bayes Classifier Implementation
export class NaiveBayes implements MLModel {
  private means: number[][] = [];
  private stds: number[][] = [];
  private priors: number[] = [];
  private classes: number[] = [];
  private smoothingFactor: number;
  private accuracy: number = 0;
  
  constructor(smoothingFactor: number = 0.5) {
    this.smoothingFactor = smoothingFactor;
  }

  train(X: number[][], y: number[]): void {
    // Get unique classes
    this.classes = Array.from(new Set(y)).sort();
    
    // Calculate prior probabilities and feature statistics per class
    for (const cls of this.classes) {
      // Filter data for current class
      const classData = X.filter((_, i) => y[i] === cls);
      
      // Prior probability
      this.priors[cls] = classData.length / X.length;
      
      // Calculate mean and std for each feature
      const classMeans: number[] = [];
      const classStds: number[] = [];
      
      // For each feature
      for (let j = 0; j < X[0].length; j++) {
        // Get values for this feature
        const featureValues = classData.map(x => x[j]);
        
        // Calculate mean
        const mean = featureValues.reduce((a, b) => a + b, 0) / featureValues.length;
        classMeans.push(mean);
        
        // Calculate standard deviation with smoothing to prevent overfitting
        const variance = featureValues.reduce((a, b) => a + Math.pow(b - mean, 2), 0) / featureValues.length;
        // Add smoothing factor to variance to prevent overfitting on small samples
        const std = Math.sqrt(variance + this.smoothingFactor) || 0.1; 
        classStds.push(std);
      }
      
      this.means[cls] = classMeans;
      this.stds[cls] = classStds;
    }
  }

  // Gaussian probability density function
  private gaussianPDF(x: number, mean: number, std: number): number {
    const exponent = Math.exp(-Math.pow(x - mean, 2) / (2 * Math.pow(std, 2)));
    return exponent / (Math.sqrt(2 * Math.PI) * std);
  }

  // Return the model's accuracy
  getAccuracy(): number {
    return this.accuracy;
  }

  // Set model accuracy after evaluation
  setAccuracy(accuracy: number): void {
    this.accuracy = accuracy;
  }

  // Evaluate the model on test data and return detailed metrics
  evaluate(xTest: number[][], yTest: number[]): { metrics: ReturnType<typeof calculateMetrics>, predictions: number[] } {
    const predictions: number[] = [];
    
    for (let i = 0; i < xTest.length; i++) {
      const prediction = this.predict(xTest[i]).prediction;
      predictions.push(prediction);
    }
    
    const metrics = calculateMetrics(predictions, yTest);
    this.accuracy = metrics.accuracy;
    
    return { metrics, predictions };
  }

  // Predict class with probability
  predict(features: number[]): { prediction: number, confidence: number } {
    const posteriors: Record<number, number> = {};
    
    for (const cls of this.classes) {
      // Start with prior probability (in log space to avoid underflow)
      let logPosterior = Math.log(this.priors[cls]);
      
      // Multiply by likelihood of each feature
      for (let j = 0; j < features.length; j++) {
        const likelihood = this.gaussianPDF(features[j], this.means[cls][j], this.stds[cls][j]);
        // Add log likelihood (equivalent to multiplying likelihoods)
        logPosterior += Math.log(likelihood + 1e-10); // Small constant to avoid log(0)
      }
      
      posteriors[cls] = logPosterior;
    }
    
    // Find class with highest posterior
    let bestClass = this.classes[0];
    let maxPosterior = posteriors[bestClass];
    
    for (const cls of this.classes) {
      if (posteriors[cls] > maxPosterior) {
        maxPosterior = posteriors[cls];
        bestClass = cls;
      }
    }
    
    // Convert log posteriors to probabilities
    const expPosteriors = Object.entries(posteriors).map(([cls, logProb]) => ({
      cls: parseInt(cls),
      prob: Math.exp(logProb)
    }));
    
    // Normalize to get probabilities
    const sumExp = expPosteriors.reduce((sum, item) => sum + item.prob, 0);
    const normalizedPosteriors = expPosteriors.map(item => ({
      cls: item.cls,
      prob: item.prob / sumExp
    }));
    
    // Find confidence (probability of predicted class)
    const confidence = normalizedPosteriors.find(item => item.cls === bestClass)?.prob || 0;
    
    return { prediction: bestClass, confidence };
  }
}
