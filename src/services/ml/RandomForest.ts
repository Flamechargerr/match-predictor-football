
import { MLModel, calculateMetrics } from './ModelInterface';

// Decision Tree Implementation for Random Forest
class DecisionTree {
  private root: any = null;
  private maxDepth: number;
  private minSamplesSplit: number = 2;
  
  constructor(maxDepth: number = 10) {
    this.maxDepth = maxDepth;
  }
  
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
export class RandomForest implements MLModel {
  private trees: DecisionTree[] = [];
  private numTrees: number;
  private maxDepth: number;
  private accuracy: number = 0;
  
  constructor(numTrees: number = 5, maxDepth: number = 3) {
    this.numTrees = numTrees;
    this.maxDepth = maxDepth;
  }
  
  train(X: number[][], y: number[]): void {
    // Bootstrap aggregating (bagging)
    for (let i = 0; i < this.numTrees; i++) {
      // Create bootstrap sample
      const { sampleX, sampleY } = this.bootstrapSample(X, y);
      
      // Train a decision tree with limited depth to prevent overfitting
      const tree = new DecisionTree(this.maxDepth);
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
    
    // Calculate confidence with a small reduction to prevent overconfidence
    const confidence = (maxVotes / this.trees.length) * 0.9;
    
    return { prediction: parseInt(bestClass), confidence };
  }
}
