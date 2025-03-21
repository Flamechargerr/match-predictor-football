
import React, { useState, useEffect } from "react";
import { Button } from "@/components/ui/button";
import { toast } from "@/components/ui/use-toast";
import FootballIcon from "@/components/FootballIcon";
import TeamStatInput from "@/components/TeamStatInput";
import PredictionCard from "@/components/PredictionCard";
import StatsRadarChart from "@/components/StatsRadarChart";
import ModelPerformanceChart from "@/components/ModelPerformanceChart";
import StatisticsIcon from "@/components/StatisticsIcon";
import ChartIcon from "@/components/ChartIcon";
import PredictIcon from "@/components/PredictIcon";
import TrophyIcon from "@/components/TrophyIcon";
import { teams } from "@/data/teams";
import { modelPerformanceData } from "@/data/models";
import { type Team, type MatchPrediction } from "@/types";
import { Separator } from "@/components/ui/separator";

const Index = () => {
  // State for home team data
  const [homeTeam, setHomeTeam] = useState<Team>({
    name: "",
    goals: "",
    shots: "",
    shotsOnTarget: "",
    redCards: "",
  });

  // State for away team data
  const [awayTeam, setAwayTeam] = useState<Team>({
    name: "",
    goals: "",
    shots: "",
    shotsOnTarget: "",
    redCards: "",
  });

  // State for predictions
  const [predictions, setPredictions] = useState<MatchPrediction[]>([]);
  
  // State for loading
  const [isLoading, setIsLoading] = useState(false);
  
  // State to control visibility of results section
  const [showResults, setShowResults] = useState(false);

  // Check if form is valid
  const isFormValid = () => {
    return (
      homeTeam.name &&
      homeTeam.goals &&
      homeTeam.shots &&
      homeTeam.shotsOnTarget &&
      homeTeam.redCards &&
      awayTeam.name &&
      awayTeam.goals &&
      awayTeam.shots &&
      awayTeam.shotsOnTarget &&
      awayTeam.redCards
    );
  };

  // Handle prediction
  const handlePredict = () => {
    if (!isFormValid()) {
      toast({
        title: "Missing information",
        description: "Please fill in all the required fields",
        variant: "destructive",
      });
      return;
    }

    if (homeTeam.name === awayTeam.name) {
      toast({
        title: "Invalid teams",
        description: "Home and Away teams must be different",
        variant: "destructive",
      });
      return;
    }

    setIsLoading(true);
    
    // Simulate API call
    setTimeout(() => {
      // Mock prediction logic
      const homeGoals = parseInt(homeTeam.goals);
      const awayGoals = parseInt(awayTeam.goals);
      
      let baseOutcome: "Home Win" | "Away Win" | "Draw";
      
      if (homeGoals > awayGoals) {
        baseOutcome = "Home Win";
      } else if (homeGoals < awayGoals) {
        baseOutcome = "Away Win";
      } else {
        // If goals are equal, consider shots on target
        const homeShotsOnTarget = parseInt(homeTeam.shotsOnTarget);
        const awayShotsOnTarget = parseInt(awayTeam.shotsOnTarget);
        
        if (homeShotsOnTarget > awayShotsOnTarget) {
          baseOutcome = "Home Win";
        } else if (homeShotsOnTarget < awayShotsOnTarget) {
          baseOutcome = "Away Win";
        } else {
          baseOutcome = "Draw";
        }
      }
      
      // Generate mock predictions from different models
      const mockPredictions: MatchPrediction[] = [
        {
          outcome: baseOutcome,
          confidence: Math.min(92.7, 65 + Math.random() * 30),
          modelName: "Logistic Regression",
          modelAccuracy: 65.7,
        },
        {
          // Higher chance of agreement but sometimes differs
          outcome: Math.random() > 0.2 ? baseOutcome : (baseOutcome === "Home Win" ? "Away Win" : "Home Win"),
          confidence: Math.min(100, 70 + Math.random() * 30),
          modelName: "Naive Bayes",
          modelAccuracy: 62.4,
        },
        {
          // Random Forest is more conservative with confidence
          outcome: Math.random() > 0.15 ? baseOutcome : "Draw",
          confidence: Math.min(84, 60 + Math.random() * 25),
          modelName: "Random Forest",
          modelAccuracy: 63.7,
        },
      ];
      
      setPredictions(mockPredictions);
      setIsLoading(false);
      setShowResults(true);
      
      // Scroll to results
      setTimeout(() => {
        document.getElementById("results")?.scrollIntoView({ 
          behavior: "smooth", 
          block: "start"
        });
      }, 100);
    }, 1500);
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-background to-background/80">
      {/* Header */}
      <header className="bg-white border-b border-border shadow-sm py-4">
        <div className="container max-w-7xl mx-auto px-4">
          <div className="flex items-center justify-between">
            <div className="flex items-center space-x-3">
              <FootballIcon className="w-10 h-10 text-primary" />
              <div>
                <h1 className="text-2xl font-bold text-gray-900">Football Match Predictor</h1>
                <p className="text-sm text-muted-foreground">
                  Enter match statistics to predict the final result with ML
                </p>
              </div>
            </div>
          </div>
        </div>
      </header>

      <main className="container max-w-7xl mx-auto px-4 py-8">
        {/* Input Form */}
        <section className="mb-10 hero-content">
          <div className="grid grid-cols-1 md:grid-cols-2 gap-6 mb-6">
            {/* Home Team Input */}
            <TeamStatInput
              teamType="home"
              teamName={homeTeam.name}
              onTeamChange={(value) => setHomeTeam({ ...homeTeam, name: value })}
              goals={homeTeam.goals}
              onGoalsChange={(value) => setHomeTeam({ ...homeTeam, goals: value })}
              shots={homeTeam.shots}
              onShotsChange={(value) => setHomeTeam({ ...homeTeam, shots: value })}
              shotsOnTarget={homeTeam.shotsOnTarget}
              onShotsOnTargetChange={(value) => setHomeTeam({ ...homeTeam, shotsOnTarget: value })}
              redCards={homeTeam.redCards}
              onRedCardsChange={(value) => setHomeTeam({ ...homeTeam, redCards: value })}
              teamOptions={teams}
              className="animate-fade-in"
            />

            {/* Away Team Input */}
            <TeamStatInput
              teamType="away"
              teamName={awayTeam.name}
              onTeamChange={(value) => setAwayTeam({ ...awayTeam, name: value })}
              goals={awayTeam.goals}
              onGoalsChange={(value) => setAwayTeam({ ...awayTeam, goals: value })}
              shots={awayTeam.shots}
              onShotsChange={(value) => setAwayTeam({ ...awayTeam, shots: value })}
              shotsOnTarget={awayTeam.shotsOnTarget}
              onShotsOnTargetChange={(value) => setAwayTeam({ ...awayTeam, shotsOnTarget: value })}
              redCards={awayTeam.redCards}
              onRedCardsChange={(value) => setAwayTeam({ ...awayTeam, redCards: value })}
              teamOptions={teams}
              className="animate-fade-in"
            />
          </div>

          {/* Submit Button */}
          <Button 
            size="lg" 
            onClick={handlePredict} 
            disabled={isLoading || !isFormValid()}
            className="w-full py-6 text-lg font-medium relative overflow-hidden group bg-gradient-to-r from-home-DEFAULT to-away-DEFAULT hover:from-home-dark hover:to-away-dark transition-all duration-300 animate-fade-in"
          >
            {isLoading ? (
              <div className="loading-dots flex space-x-2 items-center justify-center">
                <div></div>
                <div></div>
                <div></div>
              </div>
            ) : (
              <>
                <PredictIcon className="mr-2 h-5 w-5" />
                <span>PREDICT MATCH RESULT</span>
              </>
            )}
          </Button>
        </section>

        {/* Results Section */}
        {showResults && (
          <section id="results" className="animate-fade-up pt-6">
            <div className="flex items-center mb-6 space-x-3">
              <TrophyIcon className="w-6 h-6 text-primary" />
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
                      className="animate-scale-in"
                      />
                  ))}
                </div>
              </div>
            </div>

            <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
              {/* Match Statistics */}
              <div className="bg-white rounded-xl border border-border p-6 shadow-prediction animate-slide-right">
                <div className="flex items-center mb-4 space-x-2">
                  <StatisticsIcon className="w-5 h-5 text-gray-700" />
                  <h3 className="text-xl font-semibold text-gray-800">Match Statistics</h3>
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

              {/* Model Performance */}
              <div className="bg-white rounded-xl border border-border p-6 shadow-prediction animate-slide-left">
                <div className="flex items-center mb-4 space-x-2">
                  <ChartIcon className="w-5 h-5 text-gray-700" />
                  <h3 className="text-xl font-semibold text-gray-800">Model Performance</h3>
                </div>
                <div className="h-[300px]">
                  <ModelPerformanceChart models={modelPerformanceData} />
                </div>
              </div>
            </div>
          </section>
        )}
      </main>

      {/* Footer */}
      <footer className="bg-white border-t border-border py-6 mt-12">
        <div className="container max-w-7xl mx-auto px-4">
          <div className="flex flex-col md:flex-row justify-between items-center">
            <p className="text-muted-foreground text-sm">
              Football Match Predictor — Powered by Machine Learning
            </p>
            <p className="text-muted-foreground text-sm mt-2 md:mt-0">
              © {new Date().getFullYear()} Football Match Predictor Genie
            </p>
          </div>
        </div>
      </footer>
    </div>
  );
};

export default Index;
