
import React from "react";
import ConfidenceBar from "./ConfidenceBar";
import { motion } from "framer-motion";

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
          bg: "bg-home-DEFAULT",
          border: "border-home-light",
          gradient: "bg-gradient-to-br from-blue-50/50 to-white"
        };
      case "Away Win":
        return {
          text: "text-away-dark",
          bg: "bg-away-DEFAULT",
          border: "border-away-light",
          gradient: "bg-gradient-to-br from-red-50/50 to-white"
        };
      case "Draw":
        return {
          text: "text-neutral-dark",
          bg: "bg-neutral-DEFAULT",
          border: "border-neutral-light",
          gradient: "bg-gradient-to-br from-purple-50/50 to-white"
        };
      default:
        return {
          text: "text-primary",
          bg: "bg-primary",
          border: "border-primary/20",
          gradient: "bg-gradient-to-br from-blue-50/50 to-white"
        };
    }
  };

  const colors = getPredictionColor();

  return (
    <motion.div 
      initial={{ opacity: 0, scale: 0.9 }}
      animate={{ opacity: 1, scale: 1 }}
      transition={{ duration: 0.5, ease: "easeOut" }}
      className={`rounded-xl border ${colors.border} p-5 shadow-prediction ${colors.gradient} backdrop-blur-sm hover:shadow-lg transition-all duration-300 ${className}`}
    >
      <div className="flex items-center justify-between mb-4">
        <h3 className="text-lg font-medium text-gray-700">{modelName}</h3>
        <div className="text-xs px-2 py-1 rounded-full bg-blue-100 text-blue-700">
          {accuracy.toFixed(1)}% accurate
        </div>
      </div>
      
      <div className="mb-6">
        <motion.h4 
          initial={{ scale: 0.9 }}
          animate={{ scale: 1 }}
          transition={{ delay: 0.2, duration: 0.5, type: "spring" }}
          className={`text-2xl font-bold ${colors.text} mb-2`}
        >
          {prediction}
        </motion.h4>
        <div className="space-y-1">
          <div className="flex justify-between text-sm text-muted-foreground">
            <span>Confidence</span>
            <span className="font-medium">{confidence.toFixed(1)}%</span>
          </div>
          <ConfidenceBar percentage={confidence} color={colors.bg} />
        </div>
      </div>
      
      <div className="flex items-center text-sm text-gray-500">
        <span className="inline-block w-2 h-2 bg-blue-400 rounded-full mr-2"></span>
        <span>Model reliability score: <span className="font-medium">{(accuracy * 0.8 + confidence * 0.2 / 100).toFixed(1)}%</span></span>
      </div>
    </motion.div>
  );
};

export default PredictionCard;
