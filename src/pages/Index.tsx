import React, { useState, useEffect } from "react";
import { Button } from "@/components/ui/button";
import { toast } from "@/components/ui/use-toast";
import FootballIcon from "@/components/FootballIcon";
import TeamStatInput from "@/components/TeamStatInput";
import PredictionCard from "@/components/PredictionCard";
import StatsRadarChart from "@/components/StatsRadarChart";
import ModelPerformanceChart from "@/components/ModelPerformanceChart";
import TeamPlayers from "@/components/TeamPlayers";
import StatisticsIcon from "@/components/StatisticsIcon";
import ChartIcon from "@/components/ChartIcon";
import PredictIcon from "@/components/PredictIcon";
import TrophyIcon from "@/components/TrophyIcon";
import TeamFormation from "@/components/TeamFormation";
import { teams } from "@/data/teams";
import { getTeamPlayers } from "@/data/players";
import { modelPerformanceData } from "@/data/models";
import { type Team, type MatchPrediction, type Player } from "@/types";
import { Separator } from "@/components/ui/separator";
import { mlService } from "@/services/MLService";

const teamRankings: Record<string, number> = {
  "Manchester City": 3,
  "Liverpool": 5,
  "Chelsea": 8,
  "Arsenal": 10,
  "Manchester United": 12,
  "Tottenham": 14,
  "Newcastle": 23,
  "West Ham": 29,
  "Leicester": 35,
  "Everton": 42,
  "Aston Villa": 27,
  "Crystal Palace": 45,
  "Brighton": 31,
  "Wolves": 47,
  "Brentford": 52,
  "Southampton": 58,
  "Leeds": 62,
  "Burnley": 73,
  "Watford": 88,
  "Norwich": 95,
};

const teamFormations: Record<string, string> = {
  "Manchester City": "4-3-3",
  "Liverpool": "4-3-3",
  "Chelsea": "4-2-3-1",
  "Arsenal": "4-3-3",
  "Manchester United": "4-2-3-1",
  "Tottenham": "4-2-3-1",
  "Newcastle": "4-3-3",
  "West Ham": "4-2-3-1",
  "Leicester": "4-4-2",
  "Everton": "4-3-3",
  "Aston Villa": "4-3-3",
  "Crystal Palace": "4-3-3",
  "Brighton": "4-2-3-1",
  "Wolves": "4-2-3-1",
  "Brentford": "4-3-3",
  "Southampton": "4-4-2",
  "Leeds": "4-3-3",
  "Burnley": "4-4-2",
  "Watford": "4-3-3",
  "Norwich": "4-4-2",
};

const Index = () => {
  const [homeTeam, setHomeTeam] = useState<Team>({
    name: "",
    goals: "",
    shots: "",
    shotsOnTarget: "",
    redCards: "",
  });

  const [awayTeam, setAwayTeam] = useState<Team>({
    name: "",
    goals: "",
    shots: "",
    shotsOnTarget: "",
    redCards: "",
  });

  const [homePlayers, setHomePlayers] = useState<Player[]>([]);
  const [awayPlayers, setAwayPlayers] = useState<Player[]>([]);

  const [predictions, setPredictions] = useState<MatchPrediction[]>([]);
  const [isLoading, setIsLoading] = useState(false);
  const [showResults, setShowResults] = useState(false);
  const [showAdvancedView, setShowAdvancedView] = useState(false);
  const [trainingIteration, setTrainingIteration] = useState(0);
  const [trainingProgress, setTrainingProgress] = useState(0);

  useEffect(() => {
    if (homeTeam.name) {
      setHomePlayers(getTeamPlayers(homeTeam.name));
    }
  }, [homeTeam.name]);

  useEffect(() => {
    if (awayTeam.name) {
      setAwayPlayers(getTeamPlayers(awayTeam.name));
    }
  }, [awayTeam.name]);

  useEffect(() => {
    const trainingInterval = setInterval(() => {
      setTrainingIteration(prev => prev + 1);
      setTrainingProgress(prev => (prev + 5) % 100);
      mlService.improveModels();
    }, 30000);

    return () => clearInterval(trainingInterval);
  }, []);

  const isFormValid = () => {
    return (
      homeTeam.name &&
      homeTeam.goals !== "" &&
      homeTeam.shots &&
      homeTeam.shotsOnTarget &&
      homeTeam.redCards &&
      awayTeam.name &&
      awayTeam.goals !== "" &&
      awayTeam.shots &&
      awayTeam.shotsOnTarget &&
      awayTeam.redCards
    );
  };

  const handlePredict = async () => {
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
    
    try {
      const mlPredictions = await mlService.predictMatch(homeTeam, awayTeam);
      setPredictions(mlPredictions);
      setShowResults(true);
      
      setTimeout(() => {
        document.getElementById("results")?.scrollIntoView({ 
          behavior: "smooth", 
          block: "start"
        });
      }, 100);
    } catch (error) {
      console.error("Prediction error:", error);
      toast({
        title: "Prediction error",
        description: "An error occurred while making predictions",
        variant: "destructive",
      });
    } finally {
      setIsLoading(false);
    }
  };

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
              {trainingIteration > 0 && (
                <div className="text-xs text-white bg-blue-700/50 rounded-full px-3 py-1">
                  Training cycle: {trainingIteration} 
                  <span className="inline-block w-2 h-2 ml-2 bg-green-400 rounded-full animate-pulse"></span>
                </div>
              )}
              <Button 
                variant="outline" 
                onClick={() => setShowAdvancedView(!showAdvancedView)}
                className="bg-white/10 text-white border-white/30 hover:bg-white/20"
              >
                {showAdvancedView ? "Simple View" : "Advanced View"}
              </Button>
            </div>
          </div>
        </div>
      </header>

      <main className="container max-w-7xl mx-auto px-4 py-8">
        <section className="mb-10 hero-content">
          <div className="grid grid-cols-1 md:grid-cols-2 gap-6 mb-6">
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

          {showAdvancedView && (homeTeam.name || awayTeam.name) && (
            <div className="grid grid-cols-1 md:grid-cols-2 gap-6 my-6">
              {homeTeam.name && (
                <TeamFormation 
                  teamName={homeTeam.name} 
                  players={homePlayers}
                  formation={teamFormations[homeTeam.name] || "4-3-3"}
                  fifaRanking={teamRankings[homeTeam.name]}
                />
              )}
              {awayTeam.name && (
                <TeamFormation 
                  teamName={awayTeam.name} 
                  players={awayPlayers}
                  formation={teamFormations[awayTeam.name] || "4-3-3"}
                  fifaRanking={teamRankings[awayTeam.name]}
                />
              )}
            </div>
          )}

          {(homeTeam.name || awayTeam.name) && (
            <div className="grid grid-cols-1 md:grid-cols-2 gap-6 my-6">
              {homeTeam.name && (
                <TeamPlayers 
                  teamName={homeTeam.name} 
                  players={homePlayers}
                  showAll={showAdvancedView}
                  className="bg-gradient-to-br from-blue-50 to-indigo-50 p-4 rounded-lg border border-blue-100"
                />
              )}
              {awayTeam.name && (
                <TeamPlayers 
                  teamName={awayTeam.name} 
                  players={awayPlayers}
                  showAll={showAdvancedView}
                  className="bg-gradient-to-br from-red-50 to-pink-50 p-4 rounded-lg border border-red-100"
                />
              )}
            </div>
          )}

          <Button 
            size="lg" 
            onClick={handlePredict} 
            disabled={isLoading || !isFormValid()}
            className="w-full py-6 text-lg font-medium relative overflow-hidden group bg-gradient-to-r from-indigo-600 to-purple-600 hover:from-indigo-700 hover:to-purple-700 transition-all duration-300 animate-fade-in rounded-xl shadow-lg"
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

        {showResults && (
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
        )}
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

export default Index;
