
import React, { useEffect, useState } from "react";

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
  const [width, setWidth] = useState(0);
  
  useEffect(() => {
    // Animate the bar width after component mounts
    const timer = setTimeout(() => {
      setWidth(percentage);
    }, 100);
    
    return () => clearTimeout(timer);
  }, [percentage]);

  return (
    <div className={`prediction-confidence-bar ${className}`}>
      <div
        className={`h-full ${color} transition-all duration-1000 ease-out`}
        style={{ width: `${width}%` }}
      />
    </div>
  );
};

export default ConfidenceBar;
