import React, { useState } from "react";
import FootballIcon from "@/components/FootballIcon";
import TrainingCycleIndicator from "@/components/TrainingCycleIndicator";
import TrainingExplanation from "@/components/TrainingExplanation";

interface MainLayoutProps {
  children: React.ReactNode;
  trainingIteration?: number;
  trainingProgress?: number;
  showAdvancedView?: boolean;
  onToggleView?: () => void;
}

const MainLayout: React.FC<MainLayoutProps> = ({
  children,
  trainingIteration = 0,
  trainingProgress = 0,
  showAdvancedView = false,
  onToggleView,
}) => {
  const [showTrainingInfo, setShowTrainingInfo] = useState(false);

  return (
    <div className="min-h-screen bg-gradient-to-br from-indigo-50 to-purple-50">
      <header className="bg-gradient-to-r from-blue-600 to-indigo-700 shadow-md py-4">
        <div className="container max-w-7xl mx-auto px-4">
          <div className="flex items-center justify-between">
            <div className="flex items-center space-x-3">
              <div className="p-2 bg-white/20 rounded-full">
                <FootballIcon className="w-10 h-10 text-white" />
              </div>
              <div>
                <h1 className="text-2xl font-bold text-white">Football Match Predictor</h1>
                <p className="text-sm text-blue-100">
                  Enter match statistics to predict the final result with ML
                </p>
              </div>
            </div>
            <div className="flex items-center gap-3">
              {onToggleView && (
                <Button 
                  variant="outline" 
                  onClick={onToggleView}
                  className="bg-white/10 text-white border-white/30 hover:bg-white/20"
                >
                  {showAdvancedView ? "Simple View" : "Advanced View"}
                </Button>
              )}
            </div>
          </div>
        </div>
      </header>

      <main className="container max-w-7xl mx-auto px-4 py-8">
        {trainingIteration > 0 && (
          <div className="mb-8">
            <TrainingCycleIndicator 
              iteration={trainingIteration} 
              progress={trainingProgress}
              onClick={() => setShowTrainingInfo(!showTrainingInfo)}
            />
          </div>
        )}
        
        {showTrainingInfo && trainingIteration > 0 && (
          <div className="mb-8 animate-fade-down">
            <TrainingExplanation 
              trainingIteration={trainingIteration}
              modelPerformance={[
                { name: "Naive Bayes", accuracy: 0.82 + (0.001 * trainingIteration) },
                { name: "Random Forest", accuracy: 0.89 + (0.0008 * trainingIteration) },
                { name: "Logistic Regression", accuracy: 0.87 + (0.0009 * trainingIteration) }
              ]}
            />
          </div>
        )}
        {children}
      </main>

      <footer className="bg-gradient-to-r from-gray-800 to-gray-900 text-white border-t border-gray-700 py-6 mt-12">
        <div className="container max-w-7xl mx-auto px-4">
          <div className="flex flex-col md:flex-row justify-between items-center">
            <div className="flex items-center">
              <FootballIcon className="w-6 h-6 text-blue-400 mr-2" />
              <p className="text-gray-300 text-sm">
                Football Match Predictor — Powered by Machine Learning
              </p>
            </div>
            <p className="text-gray-400 text-sm mt-2 md:mt-0">
              © {new Date().getFullYear()} Football Match Predictor Genie
            </p>
          </div>
        </div>
      </footer>
    </div>
  );
};

import { Button } from "@/components/ui/button";

export default MainLayout;
