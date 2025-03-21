
import React from "react";
import ConfidenceBar from "./ConfidenceBar";

interface PredictionCardProps {
  modelName: string;
  prediction: "Home Win" | "Away Win" | "Draw";
  confidence: number;
  accuracy: number;
  className?: string;
}

const PredictionCard: React.FC<PredictionCardProps> = ({
  modelName,
  prediction,
  confidence,
  accuracy,
  className = "",
}) => {
  // Determine color based on prediction
  const getPredictionColor = () => {
    switch (prediction) {
      case "Home Win":
        return {
          text: "text-home-dark",
          bg: "bg-home-DEFAULT"
        };
      case "Away Win":
        return {
          text: "text-away-dark",
          bg: "bg-away-DEFAULT"
        };
      case "Draw":
        return {
          text: "text-neutral-dark",
          bg: "bg-neutral-DEFAULT"
        };
      default:
        return {
          text: "text-primary",
          bg: "bg-primary"
        };
    }
  };

  const colors = getPredictionColor();

  return (
    <div className={`rounded-xl border border-border p-5 shadow-prediction bg-card backdrop-blur-sm transition-all duration-300 hover:shadow-card-hover ${className}`}>
      <h3 className="text-lg font-medium text-gray-700 mb-4">{modelName}</h3>
      
      <div className="mb-6">
        <h4 className={`text-2xl font-bold ${colors.text} mb-1`}>{prediction}</h4>
        <div className="space-y-1">
          <div className="flex justify-between text-sm text-muted-foreground">
            <span>Confidence</span>
            <span className="font-medium">{confidence.toFixed(1)}%</span>
          </div>
          <ConfidenceBar percentage={confidence} color={colors.bg} />
        </div>
      </div>
      
      <div className="inline-flex items-center px-2.5 py-0.5 rounded-md text-sm bg-blue-50 text-blue-800">
        Model Accuracy: {accuracy.toFixed(1)}%
      </div>
    </div>
  );
};

export default PredictionCard;
