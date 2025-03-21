
import React from "react";
import ConfidenceBar from "./ConfidenceBar";
import { motion } from "framer-motion";
import { AlertCircle, CheckCircle2, InfoIcon } from "lucide-react";
import { Tooltip, TooltipContent, TooltipProvider, TooltipTrigger } from "@/components/ui/tooltip";

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

  // Determine confidence level indicator
  const getConfidenceIndicator = () => {
    if (confidence < 50) return <AlertCircle className="w-4 h-4 text-amber-500" />;
    if (confidence < 75) return <InfoIcon className="w-4 h-4 text-blue-500" />;
    return <CheckCircle2 className="w-4 h-4 text-green-500" />;
  };

  const colors = getPredictionColor();
  const confidenceIndicator = getConfidenceIndicator();
  const modelReliability = (accuracy * 0.8 + confidence * 0.2 / 100).toFixed(1);

  return (
    <motion.div 
      initial={{ opacity: 0, scale: 0.9 }}
      animate={{ opacity: 1, scale: 1 }}
      transition={{ duration: 0.5, ease: "easeOut" }}
      className={`rounded-xl border ${colors.border} p-5 shadow-prediction ${colors.gradient} backdrop-blur-sm hover:shadow-lg transition-all duration-300 ${className}`}
    >
      <div className="flex items-center justify-between mb-4">
        <div className="flex items-center">
          <h3 className="text-lg font-medium text-gray-700">{modelName}</h3>
          <TooltipProvider>
            <Tooltip>
              <TooltipTrigger asChild>
                <InfoIcon className="w-4 h-4 ml-2 text-muted-foreground cursor-help" />
              </TooltipTrigger>
              <TooltipContent>
                <p className="text-xs max-w-[200px]">
                  {modelName === "Naive Bayes" && "A probabilistic classifier based on Bayes' theorem with independence assumptions between features."}
                  {modelName === "Random Forest" && "An ensemble learning method that constructs multiple decision trees during training and outputs the class that is the mode of the classes."}
                  {modelName === "Logistic Regression" && "A statistical model that uses a logistic function to model a binary dependent variable."}
                </p>
              </TooltipContent>
            </Tooltip>
          </TooltipProvider>
        </div>
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
            <span className="flex items-center">
              Confidence {confidenceIndicator}
            </span>
            <span className="font-medium">{confidence.toFixed(1)}%</span>
          </div>
          <ConfidenceBar percentage={confidence} color={colors.bg} />
        </div>
      </div>
      
      <div className="flex items-center justify-between text-sm text-gray-500">
        <div className="flex items-center">
          <span className="inline-block w-2 h-2 bg-blue-400 rounded-full mr-2"></span>
          <span>Model reliability score</span>
        </div>
        <span className="font-medium">{modelReliability}%</span>
      </div>
      
      <div className="mt-4 pt-3 border-t border-gray-100 text-xs text-gray-500">
        <p>
          {modelName === "Naive Bayes" && "Feature independence assumption applied to football statistics."}
          {modelName === "Random Forest" && "Ensemble of decision trees analyzing feature importance."}
          {modelName === "Logistic Regression" && "Linear model with probability-based classification."}
        </p>
      </div>
    </motion.div>
  );
};

export default PredictionCard;
