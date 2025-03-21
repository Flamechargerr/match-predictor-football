
import * as tf from '@tensorflow/tfjs';
import { MatchPrediction, Team, ModelPerformance } from '@/types';
import { footballMatchData, trainTestSplit } from '@/data/footballMatchData';

// Import our refactored models
import { NaiveBayes } from './ml/NaiveBayes';
import { RandomForest } from './ml/RandomForest';
import { LogisticRegression } from './ml/LogisticRegression';
import { outcomeLabels, addNoise } from './ml/ModelInterface';

// Define machine learning service
class MLService {
  private naiveBayesModel: NaiveBayes | null = null;
  private randomForestModel: RandomForest | null = null;
  private logisticRegressionModel: LogisticRegression | null = null;
  private isModelTrained = false;
  private modelPerformance: ModelPerformance[] = [];

  constructor() {
    this.trainModels();
  }

  // Train the machine learning models
  private async trainModels(): Promise<void> {
    try {
      console.log("Starting model training...");
      
      // Split data into train and test sets (80/20 split)
      const { trainData, testData } = trainTestSplit(footballMatchData, 0.2, 42); // Use seed 42 for reproducible results
      
      console.log(`Training on ${trainData.length} samples, testing on ${testData.length} samples`);
      
      // Prepare training data
      const xTrain = trainData.map(d => [
        d[0], d[1], d[2], d[3], d[4], d[5], d[6], d[7]
      ]);
      const yTrain = trainData.map(d => d[8]); // Class labels: 0, 1, or 2
      
      // Prepare test data
      const xTest = testData.map(d => [
        d[0], d[1], d[2], d[3], d[4], d[5], d[6], d[7]
      ]);
      const yTest = testData.map(d => d[8]);

      // Train models in parallel
      const modelTraining = [];

      // 1. Train Naive Bayes
      this.naiveBayesModel = new NaiveBayes(0.5); // Add smoothing factor
      const nbTraining = Promise.resolve().then(() => {
        this.naiveBayesModel!.train(xTrain, yTrain);
        const { metrics } = this.naiveBayesModel!.evaluate(xTest, yTest);
        console.log(`Naive Bayes accuracy: ${(metrics.accuracy * 100).toFixed(2)}%`);
        
        // Calculate average precision across classes
        const avgPrecision = metrics.precision.reduce((a, b) => a + b, 0) / metrics.precision.length;
        
        this.modelPerformance.push({
          name: "Naive Bayes",
          accuracy: metrics.accuracy,
          precision: avgPrecision
        });
        
        return metrics;
      });
      modelTraining.push(nbTraining);

      // 2. Train Random Forest
      this.randomForestModel = new RandomForest(5, 3); // Parameters tuned for ~80% accuracy
      const rfTraining = Promise.resolve().then(() => {
        this.randomForestModel!.train(xTrain, yTrain);
        const { metrics } = this.randomForestModel!.evaluate(xTest, yTest);
        console.log(`Random Forest accuracy: ${(metrics.accuracy * 100).toFixed(2)}%`);
        
        // Calculate average precision across classes
        const avgPrecision = metrics.precision.reduce((a, b) => a + b, 0) / metrics.precision.length;
        
        this.modelPerformance.push({
          name: "Random Forest",
          accuracy: metrics.accuracy,
          precision: avgPrecision
        });
        
        return metrics;
      });
      modelTraining.push(rfTraining);

      // 3. Train Logistic Regression
      this.logisticRegressionModel = new LogisticRegression();
      const lrTraining = this.logisticRegressionModel.train(xTrain, yTrain).then(async () => {
        const { metrics } = await this.logisticRegressionModel!.evaluate(xTest, yTest);
        console.log(`Logistic Regression accuracy: ${(metrics.accuracy * 100).toFixed(2)}%`);
        
        // Calculate average precision across classes
        const avgPrecision = metrics.precision.reduce((a, b) => a + b, 0) / metrics.precision.length;
        
        this.modelPerformance.push({
          name: "Logistic Regression",
          accuracy: metrics.accuracy,
          precision: avgPrecision
        });
        
        return metrics;
      });
      modelTraining.push(lrTraining);

      // Wait for all models to finish training
      await Promise.all(modelTraining);
      
      this.isModelTrained = true;
      console.log("Models trained successfully");
    } catch (error) {
      console.error("Error training models:", error);
    }
  }

  // Get model performance for display
  public getModelPerformance(): ModelPerformance[] {
    return this.modelPerformance;
  }

  // Make a prediction based on match statistics
  public async predictMatch(homeTeam: Team, awayTeam: Team): Promise<MatchPrediction[]> {
    if (!this.isModelTrained || !this.naiveBayesModel || !this.randomForestModel || !this.logisticRegressionModel) {
      console.log("Models not ready, training now...");
      await this.trainModels();
    }

    // Prepare input data
    const inputData = [
      parseInt(homeTeam.goals),
      parseInt(awayTeam.goals),
      parseInt(homeTeam.shots),
      parseInt(awayTeam.shots),
      parseInt(homeTeam.shotsOnTarget),
      parseInt(awayTeam.shotsOnTarget),
      parseInt(homeTeam.redCards),
      parseInt(awayTeam.redCards)
    ];

    // Add small random noise to prevent overfitting
    const noisyInputData = addNoise(inputData, 0.05);

    // Get predictions from all models
    const naiveBayesPrediction = this.naiveBayesModel!.predict(noisyInputData);
    const randomForestPrediction = this.randomForestModel!.predict(noisyInputData);
    
    // Get logistic regression prediction using TensorFlow.js
    const logisticRegressionPrediction = await this.logisticRegressionModel!.predict(noisyInputData);
    
    // Format predictions with slight confidence reduction to prevent overconfidence
    const confidenceScalingFactor = 0.95; // Slightly reduce confidence
    const predictions: MatchPrediction[] = [
      {
        modelName: "Naive Bayes",
        outcome: outcomeLabels[naiveBayesPrediction.prediction],
        confidence: parseFloat((naiveBayesPrediction.confidence * 100 * confidenceScalingFactor).toFixed(1)),
        modelAccuracy: parseFloat((this.modelPerformance[0]?.accuracy * 100 || 0).toFixed(1))
      },
      {
        modelName: "Random Forest",
        outcome: outcomeLabels[randomForestPrediction.prediction],
        confidence: parseFloat((randomForestPrediction.confidence * 100 * confidenceScalingFactor).toFixed(1)),
        modelAccuracy: parseFloat((this.modelPerformance[1]?.accuracy * 100 || 0).toFixed(1))
      },
      {
        modelName: "Logistic Regression",
        outcome: outcomeLabels[logisticRegressionPrediction.prediction],
        confidence: parseFloat((logisticRegressionPrediction.confidence * 100 * confidenceScalingFactor).toFixed(1)),
        modelAccuracy: parseFloat((this.modelPerformance[2]?.accuracy * 100 || 0).toFixed(1))
      }
    ];

    return predictions;
  }
}

// Export as singleton
export const mlService = new MLService();
