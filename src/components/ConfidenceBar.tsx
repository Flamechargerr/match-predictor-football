
import React, { useEffect, useState } from "react";
import { motion } from "framer-motion";

interface ConfidenceBarProps {
  percentage: number;
  color?: string;
  className?: string;
}

const ConfidenceBar: React.FC<ConfidenceBarProps> = ({ 
  percentage,
  color = "bg-primary",
  className = ""
}) => {
  // Determine the confidence level display
  const getConfidenceLevel = () => {
    if (percentage < 50) return "Low";
    if (percentage < 75) return "Medium";
    return "High";
  };

  // Get color class based on confidence level
  const getBackgroundGradient = () => {
    if (percentage < 50) {
      return "bg-gradient-to-r from-yellow-400 to-orange-500";
    } else if (percentage < 75) {
      return "bg-gradient-to-r from-blue-400 to-blue-600";
    } else {
      return "bg-gradient-to-r from-green-400 to-green-600";
    }
  };

  const confidenceLevel = getConfidenceLevel();
  const backgroundGradient = color === "bg-primary" ? getBackgroundGradient() : color;

  return (
    <div className={`prediction-confidence-bar relative h-2 rounded-full overflow-hidden bg-gray-200 ${className}`}>
      <motion.div
        initial={{ width: 0 }}
        animate={{ width: `${percentage}%` }}
        transition={{ duration: 1, ease: "easeOut" }}
        className={`h-full ${backgroundGradient}`}
      />
      <motion.span 
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        transition={{ delay: 0.7, duration: 0.3 }}
        className="absolute top-[-18px] right-0 text-[10px] text-gray-500 font-medium"
      >
        {confidenceLevel}
      </motion.span>
    </div>
  );
};

export default ConfidenceBar;
