
import * as tf from '@tensorflow/tfjs';
import { MLModel, calculateMetrics } from './ModelInterface';

export class LogisticRegression implements MLModel {
  private model: tf.Sequential | null = null;
  private accuracy: number = 0;
  private options: {
    learningRate: number;
    epochs: number;
    batchSize: number;
    hiddenUnits: number;
    regularizationRate: number;
  };

  constructor(options?: {
    learningRate?: number;
    epochs?: number; 
    batchSize?: number;
    hiddenUnits?: number;
    regularizationRate?: number;
  }) {
    // Default hyperparameters tuned for ~80% accuracy
    this.options = {
      learningRate: options?.learningRate || 0.005,
      epochs: options?.epochs || 200,
      batchSize: options?.batchSize || 8,
      hiddenUnits: options?.hiddenUnits || 12,
      regularizationRate: options?.regularizationRate || 0.01
    };
  }

  // Return the model's accuracy
  getAccuracy(): number {
    return this.accuracy;
  }

  // Set model accuracy after evaluation
  setAccuracy(accuracy: number): void {
    this.accuracy = accuracy;
  }

  async train(X: number[][], y: number[]): Promise<void> {
    // Create a sequential model
    this.model = tf.sequential();
    
    // Add a hidden layer to improve model accuracy
    this.model.add(tf.layers.dense({
      units: this.options.hiddenUnits,
      activation: 'relu',
      inputShape: [X[0].length],
      kernelRegularizer: tf.regularizers.l2({l2: this.options.regularizationRate})
    }));
    
    // Output layer for 3-class classification
    this.model.add(tf.layers.dense({
      units: 3,
      activation: 'softmax'
    }));

    // Compile the model
    this.model.compile({
      optimizer: tf.train.adam(this.options.learningRate),
      loss: 'categoricalCrossentropy',
      metrics: ['accuracy']
    });

    // Prepare data for training
    const xs = tf.tensor2d(X, [X.length, X[0].length]);
    const ys = tf.tensor2d(y.map(label => {
      // One-hot encode the result
      if (label === 0) return [1, 0, 0]; // Home win
      if (label === 1) return [0, 1, 0]; // Draw
      return [0, 0, 1]; // Away win
    }), [y.length, 3]);

    // Train the model
    await this.model.fit(xs, ys, {
      epochs: this.options.epochs,
      batchSize: this.options.batchSize,
      shuffle: true,
      validationSplit: 0.2,
      verbose: 0
    });

    // Clean up tensors
    xs.dispose();
    ys.dispose();
  }

  // Evaluate the model on test data
  async evaluate(xTest: number[][], yTest: number[]): Promise<{ metrics: ReturnType<typeof calculateMetrics>, predictions: number[] }> {
    if (!this.model) {
      throw new Error('Model not trained yet');
    }

    // Prepare test data
    const xs = tf.tensor2d(xTest, [xTest.length, xTest[0].length]);
    
    // Get predictions
    const predictions = await this.model.predict(xs) as tf.Tensor;
    const predArray = await predictions.array() as number[][];
    
    // Convert to class indices
    const predIndices = predArray.map(pred => {
      // Get index of max value (argmax)
      return pred.indexOf(Math.max(...pred));
    });
    
    // Calculate metrics
    const metrics = calculateMetrics(predIndices, yTest);
    this.accuracy = metrics.accuracy;
    
    // Clean up tensors
    xs.dispose();
    predictions.dispose();
    
    return { metrics, predictions: predIndices };
  }

  async predict(features: number[]): Promise<{ prediction: number; confidence: number }> {
    if (!this.model) {
      throw new Error('Model not trained yet');
    }

    // Convert features to tensor
    const input = tf.tensor2d([features], [1, features.length]);
    
    // Get prediction
    const prediction = this.model.predict(input) as tf.Tensor;
    const probabilities = await prediction.array() as number[][];
    
    // Get the class with highest probability
    const pred = probabilities[0];
    const maxIndex = pred.indexOf(Math.max(...pred));
    const confidence = pred[maxIndex];
    
    // Clean up tensors
    input.dispose();
    prediction.dispose();
    
    return { prediction: maxIndex, confidence };
  }
}
