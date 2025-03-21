
import { MatchPrediction } from '@/types';

// Base model interface for all ML models
export interface MLModel {
  train(xTrain: number[][], yTrain: number[]): void;
  predict(features: number[]): { prediction: number, confidence: number };
  getAccuracy(): number;
}

// Outcome labels shared across models
export const outcomeLabels = ["Home Win", "Draw", "Away Win"] as const;

// Evaluation metrics interface
export interface ModelMetrics {
  accuracy: number;
  precision: number[];  // Precision for each class
  recall: number[];     // Recall for each class
  f1Score: number[];    // F1 score for each class
  confusionMatrix: number[][];  // Confusion matrix
}

// Helper function to calculate evaluation metrics
export function calculateMetrics(predictions: number[], actual: number[], numClasses: number = 3): ModelMetrics {
  const confusionMatrix = Array(numClasses).fill(0).map(() => Array(numClasses).fill(0));
  
  // Fill confusion matrix
  for (let i = 0; i < predictions.length; i++) {
    confusionMatrix[actual[i]][predictions[i]]++;
  }
  
  // Calculate precision, recall, and f1-score for each class
  const precision: number[] = [];
  const recall: number[] = [];
  const f1Score: number[] = [];
  
  let totalCorrect = 0;
  
  for (let i = 0; i < numClasses; i++) {
    // True positives for class i
    const tp = confusionMatrix[i][i];
    totalCorrect += tp;
    
    // Sum of predicted positives (column sum)
    const predictedPositives = confusionMatrix.reduce((sum, row) => sum + row[i], 0);
    
    // Sum of actual positives (row sum)
    const actualPositives = confusionMatrix[i].reduce((sum, cell) => sum + cell, 0);
    
    // Calculate precision and recall
    const classPrec = predictedPositives === 0 ? 0 : tp / predictedPositives;
    const classRecall = actualPositives === 0 ? 0 : tp / actualPositives;
    
    precision.push(classPrec);
    recall.push(classRecall);
    
    // Calculate F1 score
    const classF1 = classPrec === 0 || classRecall === 0 ? 
      0 : 2 * (classPrec * classRecall) / (classPrec + classRecall);
    
    f1Score.push(classF1);
  }
  
  // Overall accuracy
  const accuracy = totalCorrect / predictions.length;
  
  return {
    accuracy,
    precision,
    recall,
    f1Score,
    confusionMatrix
  };
}

// Utility function to add small random noise to prevent overfitting
export function addNoise(inputData: number[], noiseLevel: number = 0.05): number[] {
  return inputData.map(val => {
    const noise = val * (Math.random() * noiseLevel);
    return Math.max(0, val + (Math.random() > 0.5 ? noise : -noise));
  });
}
