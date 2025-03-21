
import * as tf from '@tensorflow/tfjs';
import { TrainingData, TrainedModel } from '@/types';

// Historical football match data for training
const historicalMatches: TrainingData[] = [
  { homeGoals: 2, awayGoals: 0, homeShots: 15, awayShots: 8, homeShotsOnTarget: 7, awayShotsOnTarget: 3, homeRedCards: 0, awayRedCards: 0, result: "Home Win" },
  { homeGoals: 1, awayGoals: 3, homeShots: 10, awayShots: 12, homeShotsOnTarget: 4, awayShotsOnTarget: 8, homeRedCards: 0, awayRedCards: 0, result: "Away Win" },
  { homeGoals: 0, awayGoals: 0, homeShots: 12, awayShots: 10, homeShotsOnTarget: 4, awayShotsOnTarget: 3, homeRedCards: 0, awayRedCards: 1, result: "Draw" },
  { homeGoals: 3, awayGoals: 1, homeShots: 18, awayShots: 7, homeShotsOnTarget: 9, awayShotsOnTarget: 2, homeRedCards: 0, awayRedCards: 0, result: "Home Win" },
  { homeGoals: 1, awayGoals: 1, homeShots: 13, awayShots: 15, homeShotsOnTarget: 5, awayShotsOnTarget: 6, homeRedCards: 0, awayRedCards: 0, result: "Draw" },
  { homeGoals: 0, awayGoals: 2, homeShots: 9, awayShots: 16, homeShotsOnTarget: 3, awayShotsOnTarget: 7, homeRedCards: 1, awayRedCards: 0, result: "Away Win" },
  { homeGoals: 2, awayGoals: 2, homeShots: 14, awayShots: 14, homeShotsOnTarget: 6, awayShotsOnTarget: 5, homeRedCards: 0, awayRedCards: 0, result: "Draw" },
  { homeGoals: 4, awayGoals: 0, homeShots: 20, awayShots: 6, homeShotsOnTarget: 10, awayShotsOnTarget: 1, homeRedCards: 0, awayRedCards: 1, result: "Home Win" },
  { homeGoals: 1, awayGoals: 2, homeShots: 11, awayShots: 13, homeShotsOnTarget: 4, awayShotsOnTarget: 6, homeRedCards: 0, awayRedCards: 0, result: "Away Win" },
  { homeGoals: 0, awayGoals: 1, homeShots: 12, awayShots: 11, homeShotsOnTarget: 5, awayShotsOnTarget: 4, homeRedCards: 0, awayRedCards: 0, result: "Away Win" },
  { homeGoals: 3, awayGoals: 3, homeShots: 16, awayShots: 17, homeShotsOnTarget: 8, awayShotsOnTarget: 8, homeRedCards: 0, awayRedCards: 0, result: "Draw" },
  { homeGoals: 2, awayGoals: 1, homeShots: 15, awayShots: 9, homeShotsOnTarget: 7, awayShotsOnTarget: 3, homeRedCards: 0, awayRedCards: 0, result: "Home Win" },
  { homeGoals: 0, awayGoals: 0, homeShots: 8, awayShots: 9, homeShotsOnTarget: 2, awayShotsOnTarget: 2, homeRedCards: 0, awayRedCards: 0, result: "Draw" },
  { homeGoals: 1, awayGoals: 0, homeShots: 14, awayShots: 8, homeShotsOnTarget: 6, awayShotsOnTarget: 2, homeRedCards: 0, awayRedCards: 1, result: "Home Win" },
  { homeGoals: 0, awayGoals: 3, homeShots: 7, awayShots: 19, homeShotsOnTarget: 2, awayShotsOnTarget: 9, homeRedCards: 1, awayRedCards: 0, result: "Away Win" },
  { homeGoals: 2, awayGoals: 0, homeShots: 16, awayShots: 10, homeShotsOnTarget: 8, awayShotsOnTarget: 4, homeRedCards: 0, awayRedCards: 0, result: "Home Win" },
  { homeGoals: 1, awayGoals: 1, homeShots: 12, awayShots: 11, homeShotsOnTarget: 5, awayShotsOnTarget: 4, homeRedCards: 0, awayRedCards: 0, result: "Draw" },
  { homeGoals: 0, awayGoals: 2, homeShots: 8, awayShots: 15, homeShotsOnTarget: 3, awayShotsOnTarget: 7, homeRedCards: 0, awayRedCards: 0, result: "Away Win" },
  { homeGoals: 3, awayGoals: 0, homeShots: 17, awayShots: 9, homeShotsOnTarget: 8, awayShotsOnTarget: 3, homeRedCards: 0, awayRedCards: 1, result: "Home Win" },
  { homeGoals: 2, awayGoals: 2, homeShots: 13, awayShots: 14, homeShotsOnTarget: 6, awayShotsOnTarget: 6, homeRedCards: 0, awayRedCards: 0, result: "Draw" },
];

export class MLService {
  private static models: TrainedModel[] = [];
  private static isInitialized = false;

  static async initialize(): Promise<void> {
    if (this.isInitialized) return;
    
    console.log("Initializing ML models...");
    
    // Train a Neural Network model
    const nnModel = await this.trainNeuralNetwork();
    this.models.push(nnModel);
    
    // Train a Logistic Regression model
    const lrModel = await this.trainLogisticRegression();
    this.models.push(lrModel);
    
    // Train a Random Forest-like model (ensemble of simple models)
    const rfModel = await this.trainEnsembleModel();
    this.models.push(rfModel);
    
    this.isInitialized = true;
    console.log("ML models initialized!");
  }

  static async getModels(): Promise<TrainedModel[]> {
    if (!this.isInitialized) {
      await this.initialize();
    }
    return this.models;
  }

  private static prepareData(): {
    xTrain: tf.Tensor2d;
    yTrain: tf.Tensor2d;
    xTest: tf.Tensor2d;
    yTest: tf.Tensor2d;
  } {
    // Shuffle the data
    const shuffled = [...historicalMatches].sort(() => Math.random() - 0.5);
    
    // Split into train (80%) and test (20%) sets
    const splitIndex = Math.floor(shuffled.length * 0.8);
    const trainData = shuffled.slice(0, splitIndex);
    const testData = shuffled.slice(splitIndex);
    
    // Extract features (X) and labels (y)
    const extractFeatures = (match: TrainingData) => [
      match.homeGoals, match.awayGoals, 
      match.homeShots, match.awayShots,
      match.homeShotsOnTarget, match.awayShotsOnTarget,
      match.homeRedCards, match.awayRedCards
    ];
    
    const encodeResult = (result: "Home Win" | "Away Win" | "Draw") => {
      if (result === "Home Win") return [1, 0, 0];
      if (result === "Away Win") return [0, 1, 0];
      return [0, 0, 1]; // Draw
    };
    
    const xTrain = tf.tensor2d(trainData.map(extractFeatures));
    const yTrain = tf.tensor2d(trainData.map(match => encodeResult(match.result)));
    
    const xTest = tf.tensor2d(testData.map(extractFeatures));
    const yTest = tf.tensor2d(testData.map(match => encodeResult(match.result)));
    
    return { xTrain, yTrain, xTest, yTest };
  }

  private static async trainNeuralNetwork(): Promise<TrainedModel> {
    const { xTrain, yTrain, xTest, yTest } = this.prepareData();
    
    // Create a neural network model
    const model = tf.sequential();
    model.add(tf.layers.dense({ 
      units: 16, 
      activation: 'relu', 
      inputShape: [8] 
    }));
    model.add(tf.layers.dense({ 
      units: 8, 
      activation: 'relu' 
    }));
    model.add(tf.layers.dense({ 
      units: 3, 
      activation: 'softmax' 
    }));
    
    // Compile the model
    model.compile({
      optimizer: tf.train.adam(0.01),
      loss: 'categoricalCrossentropy',
      metrics: ['accuracy']
    });
    
    // Train the model
    console.log("Training Neural Network model...");
    await model.fit(xTrain, yTrain, {
      epochs: 100,
      batchSize: 4,
      verbose: 0
    });
    
    // Evaluate the model
    const evaluation = await model.evaluate(xTest, yTest) as tf.Scalar[];
    const accuracy = (await evaluation[1].data())[0] * 100;
    console.log(`Neural Network accuracy: ${accuracy.toFixed(2)}%`);
    
    // Return the trained model interface
    return {
      name: "Neural Network",
      accuracy: parseFloat(accuracy.toFixed(1)),
      predict: async (input: number[]): Promise<{
        outcome: "Home Win" | "Away Win" | "Draw";
        confidence: number;
      }> => {
        const inputTensor = tf.tensor2d([input]);
        const prediction = model.predict(inputTensor) as tf.Tensor;
        const probs = await prediction.data();
        
        // Get the index of highest probability
        const maxProb = Math.max(...Array.from(probs));
        const maxIndex = Array.from(probs).indexOf(maxProb);
        
        let outcome: "Home Win" | "Away Win" | "Draw";
        if (maxIndex === 0) outcome = "Home Win";
        else if (maxIndex === 1) outcome = "Away Win";
        else outcome = "Draw";
        
        inputTensor.dispose();
        prediction.dispose();
        
        return {
          outcome,
          confidence: parseFloat((maxProb * 100).toFixed(1))
        };
      }
    };
  }

  private static async trainLogisticRegression(): Promise<TrainedModel> {
    const { xTrain, yTrain, xTest, yTest } = this.prepareData();
    
    // Create a logistic regression model (simple neural network)
    const model = tf.sequential();
    model.add(tf.layers.dense({ 
      units: 3, 
      activation: 'softmax', 
      inputShape: [8] 
    }));
    
    // Compile the model
    model.compile({
      optimizer: tf.train.sgd(0.1),
      loss: 'categoricalCrossentropy',
      metrics: ['accuracy']
    });
    
    // Train the model
    console.log("Training Logistic Regression model...");
    await model.fit(xTrain, yTrain, {
      epochs: 150,
      batchSize: 4,
      verbose: 0
    });
    
    // Evaluate the model
    const evaluation = await model.evaluate(xTest, yTest) as tf.Scalar[];
    const accuracy = (await evaluation[1].data())[0] * 100;
    console.log(`Logistic Regression accuracy: ${accuracy.toFixed(2)}%`);
    
    return {
      name: "Logistic Regression",
      accuracy: parseFloat(accuracy.toFixed(1)),
      predict: async (input: number[]): Promise<{
        outcome: "Home Win" | "Away Win" | "Draw";
        confidence: number;
      }> => {
        const inputTensor = tf.tensor2d([input]);
        const prediction = model.predict(inputTensor) as tf.Tensor;
        const probs = await prediction.data();
        
        const maxProb = Math.max(...Array.from(probs));
        const maxIndex = Array.from(probs).indexOf(maxProb);
        
        let outcome: "Home Win" | "Away Win" | "Draw";
        if (maxIndex === 0) outcome = "Home Win";
        else if (maxIndex === 1) outcome = "Away Win";
        else outcome = "Draw";
        
        inputTensor.dispose();
        prediction.dispose();
        
        return {
          outcome,
          confidence: parseFloat((maxProb * 100).toFixed(1))
        };
      }
    };
  }

  private static async trainEnsembleModel(): Promise<TrainedModel> {
    // Simulate a Random Forest by creating multiple simple models
    const numModels = 5;
    const models: tf.LayersModel[] = [];
    
    const { xTrain, yTrain, xTest, yTest } = this.prepareData();
    
    // Create and train multiple models
    console.log("Training Ensemble model...");
    for (let i = 0; i < numModels; i++) {
      const model = tf.sequential();
      model.add(tf.layers.dense({ 
        units: 8, 
        activation: 'relu', 
        inputShape: [8] 
      }));
      model.add(tf.layers.dense({ 
        units: 3, 
        activation: 'softmax' 
      }));
      
      model.compile({
        optimizer: tf.train.adam(0.01),
        loss: 'categoricalCrossentropy',
        metrics: ['accuracy']
      });
      
      await model.fit(xTrain, yTrain, {
        epochs: 50,
        batchSize: 4,
        verbose: 0
      });
      
      models.push(model);
    }
    
    // Evaluate the ensemble
    let correctPredictions = 0;
    const numSamples = await xTest.shape[0];
    
    for (let i = 0; i < numSamples; i++) {
      // Get the i-th test sample
      const sample = xTest.slice([i, 0], [1, 8]);
      const actual = yTest.slice([i, 0], [1, 3]);
      
      // Get predictions from all models
      const predictions = models.map(model => model.predict(sample) as tf.Tensor);
      
      // Average the predictions
      const sumPredictions = predictions.reduce(
        (acc, pred) => acc.add(pred), 
        tf.zeros([1, 3])
      );
      const avgPrediction = sumPredictions.div(tf.scalar(numModels));
      
      // Get the predicted class
      const predictedClass = avgPrediction.argMax(1);
      const actualClass = actual.argMax(1);
      
      // Check if the prediction is correct
      const isCorrect = tf.equal(predictedClass, actualClass);
      const isCorrectScalar = await isCorrect.data();
      
      if (isCorrectScalar[0]) {
        correctPredictions++;
      }
      
      // Clean up tensors
      sample.dispose();
      actual.dispose();
      predictions.forEach(p => p.dispose());
      sumPredictions.dispose();
      avgPrediction.dispose();
      predictedClass.dispose();
      actualClass.dispose();
      isCorrect.dispose();
    }
    
    const accuracy = (correctPredictions / numSamples) * 100;
    console.log(`Ensemble model accuracy: ${accuracy.toFixed(2)}%`);
    
    return {
      name: "Random Forest",
      accuracy: parseFloat(accuracy.toFixed(1)),
      predict: async (input: number[]): Promise<{
        outcome: "Home Win" | "Away Win" | "Draw";
        confidence: number;
      }> => {
        const inputTensor = tf.tensor2d([input]);
        
        // Get predictions from all models
        const predictions = models.map(model => model.predict(inputTensor) as tf.Tensor);
        
        // Average the predictions
        const sumPredictions = predictions.reduce(
          (acc, pred) => acc.add(pred), 
          tf.zeros([1, 3])
        );
        const avgPrediction = sumPredictions.div(tf.scalar(numModels));
        
        // Get probabilities
        const probs = await avgPrediction.data();
        
        // Get the index of highest probability
        const maxProb = Math.max(...Array.from(probs));
        const maxIndex = Array.from(probs).indexOf(maxProb);
        
        let outcome: "Home Win" | "Away Win" | "Draw";
        if (maxIndex === 0) outcome = "Home Win";
        else if (maxIndex === 1) outcome = "Away Win";
        else outcome = "Draw";
        
        // Clean up tensors
        inputTensor.dispose();
        predictions.forEach(p => p.dispose());
        sumPredictions.dispose();
        avgPrediction.dispose();
        
        return {
          outcome,
          confidence: parseFloat((maxProb * 100).toFixed(1))
        };
      }
    };
  }
}
