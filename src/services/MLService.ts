
import { MatchPrediction, Team, ModelPerformance } from '@/types';
import { footballMatchData } from '@/data/footballMatchData';
import { pyodideService } from './PyodideService';
import { toast } from "@/components/ui/use-toast";

// Define machine learning service
class MLService {
  private isModelTrained = false;
  private modelPerformance: ModelPerformance[] = [];
  private isTraining = false;

  constructor() {
    this.trainModels();
  }

  // Train the machine learning models using Python/scikit-learn via Pyodide
  private async trainModels(): Promise<void> {
    if (this.isTraining) return;
    
    try {
      this.isTraining = true;
      console.log("Starting model training with scikit-learn...");
      
      // Train models using Python service
      this.modelPerformance = await pyodideService.trainModels(footballMatchData);
      
      this.isModelTrained = true;
      this.isTraining = false;
      console.log("Models trained successfully with scikit-learn");
    } catch (error) {
      console.error("Error training models:", error);
      this.isTraining = false;
      toast({
        title: "Training Error",
        description: "An error occurred while training the models. Please try again.",
        variant: "destructive",
      });
    }
  }

  // Get model performance for display
  public getModelPerformance(): ModelPerformance[] {
    return this.modelPerformance.length > 0 ? 
      this.modelPerformance : 
      pyodideService.getModelPerformance();
  }

  // Make a prediction based on match statistics
  public async predictMatch(homeTeam: Team, awayTeam: Team): Promise<MatchPrediction[]> {
    if (!this.isModelTrained) {
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

    // Get predictions using Python models
    return await pyodideService.predictMatch(inputData);
  }
}

// Export as singleton
export const mlService = new MLService();
