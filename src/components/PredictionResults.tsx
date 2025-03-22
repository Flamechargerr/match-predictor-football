
import React from "react";
import { type MatchPrediction } from "@/types";
import PredictionCard from "@/components/PredictionCard";
import StatsRadarChart from "@/components/StatsRadarChart";
import ModelPerformanceChart from "@/components/ModelPerformanceChart";
import StatisticsIcon from "@/components/StatisticsIcon";
import ChartIcon from "@/components/ChartIcon";
import PredictIcon from "@/components/PredictIcon";
import TrophyIcon from "@/components/TrophyIcon";

interface PredictionResultsProps {
  predictions: MatchPrediction[];
  homeTeam: {
    name: string;
    goals: string;
    shots: string;
    shotsOnTarget: string;
    redCards: string;
  };
  awayTeam: {
    name: string;
    goals: string;
    shots: string;
    shotsOnTarget: string;
    redCards: string;
  };
  modelPerformanceData: any[];
}

const PredictionResults: React.FC<PredictionResultsProps> = ({
  predictions,
  homeTeam,
  awayTeam,
  modelPerformanceData,
}) => {
  if (predictions.length === 0) {
    return null;
  }

  return (
    <section id="results" className="animate-fade-up pt-6">
      <div className="flex items-center mb-6 space-x-3">
        <div className="p-2 bg-gradient-to-r from-yellow-400 to-amber-500 rounded-full">
          <TrophyIcon className="w-6 h-6 text-white" />
        </div>
        <h2 className="text-2xl font-bold text-gray-900">Match Analysis & Predictions</h2>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-8 mb-10">
        <div className="lg:col-span-3">
          <div className="flex items-center mb-4 space-x-2">
            <PredictIcon className="w-5 h-5 text-gray-700" />
            <h3 className="text-xl font-semibold text-gray-800">Model Predictions</h3>
          </div>
          <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
            {predictions.map((prediction, index) => (
              <PredictionCard
                key={index}
                modelName={prediction.modelName}
                prediction={prediction.outcome}
                confidence={prediction.confidence}
                accuracy={prediction.modelAccuracy}
                className="animate-scale-in shadow-xl"
              />
            ))}
          </div>
        </div>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
        <div className="bg-gradient-to-br from-white to-blue-50 rounded-xl border border-blue-100 p-6 shadow-prediction animate-slide-right">
          <div className="flex items-center mb-4 space-x-2">
            <div className="p-1.5 bg-blue-100 rounded-full">
              <StatisticsIcon className="w-5 h-5 text-blue-700" />
            </div>
            <h3 className="text-xl font-semibold text-blue-900">Match Statistics</h3>
          </div>
          {homeTeam.name && awayTeam.name && (
            <StatsRadarChart
              data={{
                homeTeam: {
                  name: homeTeam.name,
                  goals: parseInt(homeTeam.goals),
                  shots: parseInt(homeTeam.shots),
                  shotsOnTarget: parseInt(homeTeam.shotsOnTarget),
                  redCards: parseInt(homeTeam.redCards),
                },
                awayTeam: {
                  name: awayTeam.name,
                  goals: parseInt(awayTeam.goals),
                  shots: parseInt(awayTeam.shots),
                  shotsOnTarget: parseInt(awayTeam.shotsOnTarget),
                  redCards: parseInt(awayTeam.redCards),
                },
              }}
            />
          )}
        </div>

        <div className="bg-gradient-to-br from-white to-purple-50 rounded-xl border border-purple-100 p-6 shadow-prediction animate-slide-left">
          <div className="flex items-center mb-4 space-x-2">
            <div className="p-1.5 bg-purple-100 rounded-full">
              <ChartIcon className="w-5 h-5 text-purple-700" />
            </div>
            <h3 className="text-xl font-semibold text-purple-900">Model Performance</h3>
          </div>
          <div className="h-[300px]">
            <ModelPerformanceChart models={modelPerformanceData} />
          </div>
        </div>
      </div>
    </section>
  );
};

export default PredictionResults;
