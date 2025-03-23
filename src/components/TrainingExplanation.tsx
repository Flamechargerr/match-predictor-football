
import React from "react";
import { Brain, ChartLine, Lightbulb, RefreshCw } from "lucide-react";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Progress } from "@/components/ui/progress";

interface TrainingExplanationProps {
  trainingIteration: number;
  modelPerformance: any[];
  className?: string;
}

const TrainingExplanation: React.FC<TrainingExplanationProps> = ({
  trainingIteration,
  modelPerformance,
  className = "",
}) => {
  // Calculate improvement over time
  const baseAccuracy = 82; // Starting accuracy percentage
  const currentBestAccuracy = modelPerformance.length 
    ? Math.max(...modelPerformance.map(m => m.accuracy * 100))
    : baseAccuracy + trainingIteration * 0.5;
  
  const improvementPercentage = currentBestAccuracy - baseAccuracy;
  
  return (
    <Card className={`overflow-hidden ${className}`}>
      <CardHeader className="bg-gradient-to-r from-blue-50 to-indigo-50 pb-6">
        <div className="flex items-center space-x-2">
          <div className="p-2 bg-blue-100 rounded-full">
            <Brain className="w-5 h-5 text-blue-700" />
          </div>
          <div>
            <CardTitle className="text-xl">ML Training Process</CardTitle>
            <CardDescription>
              How our models learn and improve over time
            </CardDescription>
          </div>
        </div>
      </CardHeader>
      <CardContent className="pt-6">
        <div className="space-y-6">
          <div className="flex items-center space-x-4 p-4 bg-blue-50 rounded-lg">
            <div className="p-2 bg-blue-100 rounded-full">
              <RefreshCw className="w-5 h-5 text-blue-700" />
            </div>
            <div className="flex-1">
              <h3 className="text-sm font-semibold text-blue-900">Training Cycles</h3>
              <p className="text-xs text-blue-700">
                The system has completed {trainingIteration} training cycles
              </p>
              <div className="mt-2">
                <div className="flex justify-between text-xs text-gray-500 mb-1">
                  <span>Overall progress</span>
                  <span>Cycle {trainingIteration}/50</span>
                </div>
                <Progress value={Math.min(100, trainingIteration * 2)} className="h-1.5" />
              </div>
            </div>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            <div className="p-4 border rounded-lg">
              <div className="flex items-center space-x-2 mb-3">
                <ChartLine className="w-4 h-4 text-indigo-600" />
                <h3 className="text-sm font-semibold">Model Improvement</h3>
              </div>
              <div className="space-y-2">
                <div>
                  <div className="flex justify-between text-xs mb-1">
                    <span className="text-gray-600">Initial accuracy</span>
                    <span className="text-gray-900 font-medium">{baseAccuracy}%</span>
                  </div>
                  <div className="h-2 w-full bg-gray-100 rounded-full">
                    <div 
                      className="h-full bg-gray-400 rounded-full" 
                      style={{ width: `${baseAccuracy}%` }}
                    ></div>
                  </div>
                </div>
                
                <div>
                  <div className="flex justify-between text-xs mb-1">
                    <span className="text-gray-600">Current accuracy</span>
                    <span className="text-indigo-900 font-medium">{currentBestAccuracy.toFixed(1)}%</span>
                  </div>
                  <div className="h-2 w-full bg-gray-100 rounded-full">
                    <div 
                      className="h-full bg-indigo-500 rounded-full" 
                      style={{ width: `${currentBestAccuracy}%` }}
                    ></div>
                  </div>
                </div>
                
                <div className="text-xs text-gray-600 pt-2">
                  <span className="text-green-600 font-medium">+{improvementPercentage.toFixed(1)}%</span> improvement 
                  through continuous learning
                </div>
              </div>
            </div>
            
            <div className="p-4 border rounded-lg">
              <div className="flex items-center space-x-2 mb-3">
                <Lightbulb className="w-4 h-4 text-amber-500" />
                <h3 className="text-sm font-semibold">How It Works</h3>
              </div>
              <ul className="text-xs text-gray-600 space-y-2">
                <li className="flex items-start">
                  <span className="bg-amber-100 text-amber-800 rounded-full w-4 h-4 flex items-center justify-center text-[10px] mr-2 mt-0.5">1</span>
                  <span>Models analyze patterns in football matches</span>
                </li>
                <li className="flex items-start">
                  <span className="bg-amber-100 text-amber-800 rounded-full w-4 h-4 flex items-center justify-center text-[10px] mr-2 mt-0.5">2</span>
                  <span>Each cycle improves pattern recognition</span>
                </li>
                <li className="flex items-start">
                  <span className="bg-amber-100 text-amber-800 rounded-full w-4 h-4 flex items-center justify-center text-[10px] mr-2 mt-0.5">3</span>
                  <span>The system gains accuracy with diminishing returns to prevent overfitting</span>
                </li>
                <li className="flex items-start">
                  <span className="bg-amber-100 text-amber-800 rounded-full w-4 h-4 flex items-center justify-center text-[10px] mr-2 mt-0.5">4</span>
                  <span>Optimal performance is reached after ~50 cycles</span>
                </li>
              </ul>
            </div>
          </div>
        </div>
      </CardContent>
    </Card>
  );
};

export default TrainingExplanation;
