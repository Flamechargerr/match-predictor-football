
import React from "react";
import { Brain, RefreshCw, ChartLine, Lightbulb } from "lucide-react";
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
                    w-80 bg-white text-gray-800 rounded-lg shadow-lg p-4 pointer-events-none">
        <div className="absolute bottom-0 left-1/2 transform -translate-x-1/2 translate-y-1/2
                       w-3 h-3 bg-white rotate-45"></div>
        <div className="flex items-center mb-2">
          <Brain className="w-5 h-5 text-indigo-600 mr-2" />
          <h4 className="font-semibold text-sm">Model Training Cycle</h4>
        </div>
        <p className="text-xs text-gray-600 mb-2">
          The ML models are continuously training in the background, improving with each iteration.
          Training iteration {iteration} is now complete.
        </p>
        <div className="mb-2">
          <div className="flex justify-between text-xs text-gray-500 mb-1">
            <span>Training progress</span>
            <span>{Math.min(100, progress.toFixed(0))}%</span>
          </div>
          <Progress value={progress} className="h-1.5" />
        </div>
        <div className="flex items-start space-x-2 mt-3">
          <ChartLine className="w-4 h-4 text-green-600 mt-0.5" />
          <div className="flex-1">
            <h5 className="text-xs font-medium">Model improvement</h5>
            <p className="text-xs text-gray-500">
              Each cycle learns from previous results, improving accuracy and confidence in predictions.
            </p>
          </div>
        </div>
      </div>
    </div>
  );
};

export default TrainingCycleIndicator;
