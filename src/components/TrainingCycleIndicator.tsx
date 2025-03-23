import React, { useState, useEffect } from "react";
import { Brain, RefreshCw, ChartLine, Lightbulb, Terminal } from "lucide-react";
import { Progress } from "@/components/ui/progress";

interface TrainingCycleIndicatorProps {
  iteration: number;
  progress: number;
  onClick?: () => void;
}

const TrainingCycleIndicator: React.FC<TrainingCycleIndicatorProps> = ({
  iteration,
  progress,
  onClick,
}) => {
  const [trainingLogs, setTrainingLogs] = useState<Array<{iteration: number, size: number, accuracy: number}>>([]);
  
  useEffect(() => {
    // Generate training logs based on current iteration
    if (iteration > 0 && (iteration % 1 === 0 || trainingLogs.length === 0)) {
      // Create a pattern of increasing dataset size and accuracy
      const baseSize = 100;
      const newSize = baseSize + (iteration * 500);
      
      // Calculate a realistic accuracy that improves but plateaus
      const baseAccuracy = 0.72;
      const maxAccuracy = 0.95;
      const improvementRate = 0.02;
      
      // Accuracy increases with diminishing returns
      const newAccuracy = Math.min(
        maxAccuracy, 
        baseAccuracy + (1 - Math.exp(-iteration * improvementRate)) * (maxAccuracy - baseAccuracy)
      );
      
      // Add this new log to our collection
      const newLog = {
        iteration,
        size: newSize,
        accuracy: newAccuracy
      };
      
      // Keep last 10 logs for display
      setTrainingLogs(prev => {
        const updatedLogs = [...prev, newLog];
        return updatedLogs.slice(-10);
      });
    }
  }, [iteration]);

  return (
    <div 
      className="relative group"
      onClick={onClick}
    >
      <div className="flex items-center bg-gradient-to-r from-blue-700 to-indigo-800 text-white rounded-full px-3 py-1.5 cursor-pointer">
        <RefreshCw className="w-4 h-4 mr-2 animate-spin-slow" />
        <span className="text-xs font-medium">
          Training cycle: {iteration}
        </span>
        <div className="flex ml-2 space-x-1">
          {[...Array(3)].map((_, i) => (
            <div 
              key={i} 
              className={`w-1.5 h-1.5 rounded-full ${
                progress > (i + 1) * 33 
                  ? "bg-green-400" 
                  : "bg-white/30"
              }`}
            />
          ))}
        </div>
      </div>
      
      <div className="absolute z-50 -top-2 left-1/2 transform -translate-x-1/2 -translate-y-full 
                    opacity-0 group-hover:opacity-100 transition-opacity duration-200
                    w-96 bg-gray-900 text-gray-200 rounded-lg shadow-lg p-4 pointer-events-none">
        <div className="absolute bottom-0 left-1/2 transform -translate-x-1/2 translate-y-1/2
                       w-3 h-3 bg-gray-900 rotate-45"></div>
        <div className="flex items-center mb-2">
          <Brain className="w-5 h-5 text-indigo-400 mr-2" />
          <h4 className="font-semibold text-sm">Model Training Cycle</h4>
        </div>
        <p className="text-xs text-gray-400 mb-2">
          The ML models are continuously training in the background, improving with each iteration.
          Training iteration {iteration} is now complete.
        </p>
        <div className="mb-2">
          <div className="flex justify-between text-xs text-gray-400 mb-1">
            <span>Training progress</span>
            <span>{Math.min(100, Math.round(progress))}%</span>
          </div>
          <Progress value={progress} className="h-1.5 bg-gray-700" />
        </div>
        
        {/* Terminal-like console showing training logs */}
        <div className="mt-3 bg-black rounded-md p-2 font-mono text-xs overflow-hidden">
          <div className="flex items-center mb-2">
            <Terminal className="w-4 h-4 text-green-500 mr-2" />
            <span className="text-green-500">Training logs</span>
          </div>
          <div className="h-40 overflow-y-auto scrollbar-thin scrollbar-thumb-gray-700 scrollbar-track-gray-900">
            {trainingLogs.map((log, index) => (
              <div key={index} className="text-[10px] leading-tight">
                <span className="text-pink-500">Training size: </span>
                <span className="text-yellow-400">{log.size}</span>
                <span className="text-pink-500">, Accuracy: </span>
                <span className="text-yellow-400">{log.accuracy.toFixed(2)}</span>
              </div>
            ))}
          </div>
        </div>
        
        <div className="flex items-start space-x-2 mt-3">
          <ChartLine className="w-4 h-4 text-green-500 mt-0.5" />
          <div className="flex-1">
            <h5 className="text-xs font-medium">Model improvement</h5>
            <p className="text-[10px] text-gray-400">
              Each cycle learns from previous results, improving accuracy and confidence in predictions.
            </p>
          </div>
        </div>
      </div>
    </div>
  );
};

export default TrainingCycleIndicator;
