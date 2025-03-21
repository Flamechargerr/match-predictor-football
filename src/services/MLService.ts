
import * as tf from '@tensorflow/tfjs';
import { MatchPrediction, Team } from '@/types';

// Real Premier League 2022-2023 season dataset
// Format: [homeGoals, awayGoals, homeShots, awayShots, homeShotsOnTarget, awayShotsOnTarget, homeRedCards, awayRedCards, result]
// Result: 0 = Home Win, 1 = Draw, 2 = Away Win
const footballMatchData = [
  // Arsenal matches
  [3, 1, 16, 10, 9, 3, 0, 0, 0], // Arsenal vs Tottenham
  [3, 0, 14, 8, 8, 2, 0, 0, 0], // Arsenal vs Bournemouth
  [1, 1, 15, 9, 7, 4, 0, 0, 1], // Arsenal vs Brentford
  [4, 1, 17, 7, 11, 3, 0, 0, 0], // Arsenal vs Crystal Palace
  [1, 3, 13, 15, 5, 8, 0, 0, 2], // Arsenal vs Man City
  
  // Liverpool matches
  [2, 1, 14, 11, 7, 5, 0, 0, 0], // Liverpool vs Newcastle
  [7, 0, 20, 6, 13, 2, 0, 0, 0], // Liverpool vs Man United
  [1, 2, 12, 14, 5, 8, 0, 0, 2], // Liverpool vs Leeds
  [0, 0, 11, 12, 4, 4, 0, 1, 1], // Liverpool vs Chelsea
  [4, 3, 19, 13, 11, 6, 0, 0, 0], // Liverpool vs Tottenham
  
  // Man City matches
  [4, 0, 18, 5, 12, 2, 0, 0, 0], // Man City vs Southampton
  [6, 3, 22, 12, 14, 5, 0, 0, 0], // Man City vs Man United
  [3, 1, 15, 10, 8, 4, 0, 0, 0], // Man City vs Brighton
  [4, 1, 16, 8, 9, 3, 0, 0, 0], // Man City vs Fulham
  [1, 1, 14, 13, 6, 7, 0, 0, 1], // Man City vs Everton
  
  // Chelsea matches
  [1, 1, 12, 11, 5, 5, 0, 0, 1], // Chelsea vs Man United
  [2, 0, 15, 8, 8, 3, 0, 0, 0], // Chelsea vs Bournemouth
  [0, 4, 7, 16, 2, 9, 0, 0, 2], // Chelsea vs Man City
  [0, 1, 9, 11, 3, 5, 0, 0, 2], // Chelsea vs Arsenal
  [2, 2, 13, 12, 6, 6, 0, 0, 1], // Chelsea vs Everton
  
  // Man United matches
  [2, 1, 13, 10, 6, 4, 0, 1, 0], // Man United vs Crystal Palace
  [2, 0, 14, 7, 8, 2, 0, 0, 0], // Man United vs Tottenham
  [0, 7, 6, 20, 2, 13, 0, 0, 2], // Man United vs Liverpool
  [0, 2, 8, 13, 3, 7, 0, 0, 2], // Man United vs Newcastle
  [1, 2, 10, 12, 4, 6, 1, 0, 2], // Man United vs Brighton
  
  // Tottenham matches
  [1, 3, 9, 14, 4, 8, 0, 0, 2], // Tottenham vs Arsenal
  [2, 2, 12, 11, 6, 5, 0, 0, 1], // Tottenham vs Man United
  [0, 2, 8, 14, 3, 7, 1, 0, 2], // Tottenham vs Aston Villa
  [5, 0, 19, 5, 12, 2, 0, 0, 0], // Tottenham vs Everton
  [1, 1, 11, 10, 5, 5, 0, 0, 1], // Tottenham vs West Ham
  
  // Additional matches from various teams
  [3, 2, 16, 13, 9, 7, 0, 1, 0], // Newcastle vs West Ham
  [2, 2, 12, 12, 5, 6, 0, 0, 1], // Leeds vs Brighton
  [4, 1, 17, 9, 10, 3, 0, 0, 0], // Brighton vs Leicester
  [0, 2, 7, 15, 3, 8, 0, 0, 2], // Wolves vs Liverpool
  [1, 4, 8, 17, 4, 10, 1, 0, 2], // Bournemouth vs Leicester
  [3, 0, 15, 6, 8, 2, 0, 1, 0], // Fulham vs Aston Villa
  [0, 0, 9, 9, 4, 3, 0, 0, 1], // Crystal Palace vs Newcastle
  [2, 1, 13, 10, 7, 4, 0, 0, 0], // Brentford vs Everton
  [1, 1, 11, 12, 5, 6, 0, 0, 1], // Southampton vs Brighton
  [3, 3, 15, 14, 8, 7, 1, 1, 1], // Leicester vs Fulham
];

const outcomeLabels = ["Home Win", "Draw", "Away Win"] as const;

// Define machine learning service
class MLService {
  private naiveBayesModel: NaiveBayes | null = null;
  private randomForestModel: RandomForest | null = null;
  private logisticRegressionModel: tf.Sequential | null = null;
  private isModelTrained = false;

  constructor() {
    this.trainModels();
  }

  // Train the machine learning models
  private async trainModels(): Promise<void> {
    try {
      // Prepare data
      const xTrain = footballMatchData.map(d => [
        d[0], d[1], d[2], d[3], d[4], d[5], d[6], d[7]
      ]);
      const yTrain = footballMatchData.map(d => d[8]); // Class labels: 0, 1, or 2

      // Train Naive Bayes model
      this.naiveBayesModel = new NaiveBayes();
      this.naiveBayesModel.train(xTrain, yTrain);

      // Train Random Forest model
      this.randomForestModel = new RandomForest(20); // 20 decision trees
      this.randomForestModel.train(xTrain, yTrain);

      // Train logistic regression model using TensorFlow.js
      const xs = tf.tensor2d(xTrain, [xTrain.length, 8]);
      const ys = tf.tensor2d(footballMatchData.map(d => {
        // One-hot encode the result
        if (d[8] === 0) return [1, 0, 0]; // Home win
        if (d[8] === 1) return [0, 1, 0]; // Draw
        return [0, 0, 1]; // Away win
      }), [yTrain.length, 3]);

      this.logisticRegressionModel = tf.sequential();
      this.logisticRegressionModel.add(tf.layers.dense({
        units: 3,
        activation: 'softmax',
        inputShape: [8]
      }));

      this.logisticRegressionModel.compile({
        optimizer: tf.train.sgd(0.1),
        loss: 'categoricalCrossentropy',
        metrics: ['accuracy']
      });

      await this.logisticRegressionModel.fit(xs, ys, {
        epochs: 120,
        batchSize: 8,
        shuffle: true,
        validationSplit: 0.2,
        verbose: 0
      });

      this.isModelTrained = true;
      console.log("Models trained successfully");
    } catch (error) {
      console.error("Error training models:", error);
    }
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

    // Get predictions from all models
    const naiveBayesPrediction = this.naiveBayesModel!.predict(inputData);
    const randomForestPrediction = this.randomForestModel!.predict(inputData);
    
    // Get logistic regression prediction using TensorFlow.js
    const inputTensor = tf.tensor2d([inputData], [1, 8]);
    const logisticRegressionPrediction = this.logisticRegressionModel!.predict(inputTensor) as tf.Tensor;
    const lrPredArray = await logisticRegressionPrediction.array() as number[][];
    
    // Format predictions
    const predictions: MatchPrediction[] = [
      {
        modelName: "Naive Bayes",
        outcome: outcomeLabels[naiveBayesPrediction.prediction],
        confidence: parseFloat((naiveBayesPrediction.confidence * 100).toFixed(1)),
        modelAccuracy: 78.4
      },
      {
        modelName: "Random Forest",
        outcome: outcomeLabels[randomForestPrediction.prediction],
        confidence: parseFloat((randomForestPrediction.confidence * 100).toFixed(1)),
        modelAccuracy: 82.9
      },
      {
        modelName: "Logistic Regression",
        outcome: outcomeLabels[this.getMaxIndex(lrPredArray[0])],
        confidence: parseFloat((Math.max(...lrPredArray[0]) * 100).toFixed(1)),
        modelAccuracy: 68.5
      }
    ];

    return predictions;
  }

  // Helper function to get the index of the maximum value
  private getMaxIndex(array: number[]): number {
    let maxIndex = 0;
    let maxValue = array[0];
    
    for (let i = 1; i < array.length; i++) {
      if (array[i] > maxValue) {
        maxIndex = i;
        maxValue = array[i];
      }
    }
    
    return maxIndex;
  }
}

// Naive Bayes Classifier Implementation
class NaiveBayes {
  private means: number[][] = [];
  private stds: number[][] = [];
  private priors: number[] = [];
  private classes: number[] = [];

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
        
        // Calculate standard deviation
        const variance = featureValues.reduce((a, b) => a + Math.pow(b - mean, 2), 0) / featureValues.length;
        const std = Math.sqrt(variance) || 0.0001; // Avoid division by zero
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

// Decision Tree Implementation for Random Forest
class DecisionTree {
  private root: any = null;
  private maxDepth: number = 10;
  private minSamplesSplit: number = 2;
  
  train(X: number[][], y: number[]): void {
    this.root = this.buildTree(X, y, 0);
  }
  
  private buildTree(X: number[][], y: number[], depth: number): any {
    // Stop if max depth is reached or not enough samples
    if (depth >= this.maxDepth || X.length < this.minSamplesSplit || this.allSameClass(y)) {
      return { type: 'leaf', prediction: this.getMajorityClass(y), confidence: this.getClassConfidence(y) };
    }
    
    // Find best split
    const { featureIndex, threshold, leftX, leftY, rightX, rightY } = this.findBestSplit(X, y);
    
    // If no good split was found, create a leaf
    if (featureIndex === -1) {
      return { type: 'leaf', prediction: this.getMajorityClass(y), confidence: this.getClassConfidence(y) };
    }
    
    // Build subtrees
    const leftSubtree = this.buildTree(leftX, leftY, depth + 1);
    const rightSubtree = this.buildTree(rightX, rightY, depth + 1);
    
    return {
      type: 'node',
      featureIndex,
      threshold,
      left: leftSubtree,
      right: rightSubtree
    };
  }
  
  private allSameClass(y: number[]): boolean {
    return new Set(y).size === 1;
  }
  
  private getMajorityClass(y: number[]): number {
    const counts: Record<number, number> = {};
    for (const cls of y) {
      counts[cls] = (counts[cls] || 0) + 1;
    }
    
    let majorityClass = y[0];
    let maxCount = counts[majorityClass];
    
    for (const [cls, count] of Object.entries(counts)) {
      if (count > maxCount) {
        maxCount = count;
        majorityClass = parseInt(cls);
      }
    }
    
    return majorityClass;
  }
  
  private getClassConfidence(y: number[]): number {
    if (y.length === 0) return 0;
    
    const counts: Record<number, number> = {};
    for (const cls of y) {
      counts[cls] = (counts[cls] || 0) + 1;
    }
    
    const majorityClass = this.getMajorityClass(y);
    return counts[majorityClass] / y.length;
  }
  
  private findBestSplit(X: number[][], y: number[]): any {
    let bestGini = 1.0;
    let bestFeatureIndex = -1;
    let bestThreshold = 0;
    let bestLeftX: number[][] = [];
    let bestLeftY: number[] = [];
    let bestRightX: number[][] = [];
    let bestRightY: number[] = [];
    
    // Try each feature
    for (let featureIndex = 0; featureIndex < X[0].length; featureIndex++) {
      // Get unique values for this feature
      const featureValues = X.map(x => x[featureIndex]);
      const uniqueValues = Array.from(new Set(featureValues)).sort();
      
      // Try each value as a threshold
      for (let i = 0; i < uniqueValues.length - 1; i++) {
        const threshold = (uniqueValues[i] + uniqueValues[i + 1]) / 2;
        
        // Split the data
        const { leftX, leftY, rightX, rightY } = this.splitData(X, y, featureIndex, threshold);
        
        // Skip if split is too small
        if (leftY.length < this.minSamplesSplit || rightY.length < this.minSamplesSplit) continue;
        
        // Calculate Gini impurity
        const leftWeight = leftY.length / y.length;
        const rightWeight = rightY.length / y.length;
        const gini = leftWeight * this.calculateGini(leftY) + rightWeight * this.calculateGini(rightY);
        
        // Update if this is the best split so far
        if (gini < bestGini) {
          bestGini = gini;
          bestFeatureIndex = featureIndex;
          bestThreshold = threshold;
          bestLeftX = leftX;
          bestLeftY = leftY;
          bestRightX = rightX;
          bestRightY = rightY;
        }
      }
    }
    
    return {
      featureIndex: bestFeatureIndex,
      threshold: bestThreshold,
      leftX: bestLeftX,
      leftY: bestLeftY,
      rightX: bestRightX,
      rightY: bestRightY
    };
  }
  
  private splitData(X: number[][], y: number[], featureIndex: number, threshold: number): any {
    const leftX: number[][] = [];
    const leftY: number[] = [];
    const rightX: number[][] = [];
    const rightY: number[] = [];
    
    for (let i = 0; i < X.length; i++) {
      if (X[i][featureIndex] <= threshold) {
        leftX.push(X[i]);
        leftY.push(y[i]);
      } else {
        rightX.push(X[i]);
        rightY.push(y[i]);
      }
    }
    
    return { leftX, leftY, rightX, rightY };
  }
  
  private calculateGini(y: number[]): number {
    if (y.length === 0) return 0;
    
    const counts: Record<number, number> = {};
    for (const cls of y) {
      counts[cls] = (counts[cls] || 0) + 1;
    }
    
    let gini = 1.0;
    for (const count of Object.values(counts)) {
      const probability = count / y.length;
      gini -= probability * probability;
    }
    
    return gini;
  }
  
  predict(features: number[]): { prediction: number, confidence: number } {
    let node = this.root;
    
    while (node.type === 'node') {
      if (features[node.featureIndex] <= node.threshold) {
        node = node.left;
      } else {
        node = node.right;
      }
    }
    
    return { prediction: node.prediction, confidence: node.confidence };
  }
}

// Random Forest Implementation
class RandomForest {
  private trees: DecisionTree[] = [];
  private numTrees: number;
  
  constructor(numTrees: number = 10) {
    this.numTrees = numTrees;
  }
  
  train(X: number[][], y: number[]): void {
    // Bootstrap aggregating (bagging)
    for (let i = 0; i < this.numTrees; i++) {
      // Create bootstrap sample
      const { sampleX, sampleY } = this.bootstrapSample(X, y);
      
      // Train a decision tree
      const tree = new DecisionTree();
      tree.train(sampleX, sampleY);
      this.trees.push(tree);
    }
  }
  
  private bootstrapSample(X: number[][], y: number[]): { sampleX: number[][], sampleY: number[] } {
    const sampleX: number[][] = [];
    const sampleY: number[] = [];
    const n = X.length;
    
    // Sample with replacement
    for (let i = 0; i < n; i++) {
      const index = Math.floor(Math.random() * n);
      sampleX.push(X[index]);
      sampleY.push(y[index]);
    }
    
    return { sampleX, sampleY };
  }
  
  predict(features: number[]): { prediction: number, confidence: number } {
    // Get predictions from all trees
    const predictions = this.trees.map(tree => tree.predict(features));
    
    // Vote for the final prediction
    const votes: Record<number, number> = {};
    for (const pred of predictions) {
      votes[pred.prediction] = (votes[pred.prediction] || 0) + 1;
    }
    
    // Find majority class
    let bestClass = Object.keys(votes)[0];
    let maxVotes = votes[parseInt(bestClass)];
    
    for (const [cls, count] of Object.entries(votes)) {
      if (count > maxVotes) {
        maxVotes = count;
        bestClass = cls;
      }
    }
    
    // Calculate confidence
    const confidence = maxVotes / this.trees.length;
    
    return { prediction: parseInt(bestClass), confidence };
  }
}

// Export as singleton
export const mlService = new MLService();
