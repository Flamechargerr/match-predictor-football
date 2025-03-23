
import React, { useState, useEffect } from "react";
import { toast } from "@/components/ui/use-toast";
import MainLayout from "@/components/layout/MainLayout";
import TeamInputForm from "@/components/TeamInputForm";
import PredictionResults from "@/components/PredictionResults";
import { teams } from "@/data/teams";
import { getTeamPlayers } from "@/data/players";
import { modelPerformanceData } from "@/data/models";
import { teamFormations, teamRankings } from "@/constants/teamData";
import { type Team, type MatchPrediction, type Player } from "@/types";
import { mlService } from "@/services/MLService";

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
      // Make progress more visually interesting by making it cycle
      setTrainingProgress(prev => (prev + 33) % 100);
      mlService.improveModels();
    }, 30000);

    return () => clearInterval(trainingInterval);
  }, []);

  const handlePredict = async () => {
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
    <MainLayout
      trainingIteration={trainingIteration}
      trainingProgress={trainingProgress}
      showAdvancedView={showAdvancedView}
      onToggleView={() => setShowAdvancedView(!showAdvancedView)}
    >
      <TeamInputForm
        homeTeam={homeTeam}
        setHomeTeam={setHomeTeam}
        awayTeam={awayTeam}
        setAwayTeam={setAwayTeam}
        homePlayers={homePlayers}
        awayPlayers={awayPlayers}
        showAdvancedView={showAdvancedView}
        teamFormations={teamFormations}
        teamRankings={teamRankings}
        isLoading={isLoading}
        onPredict={handlePredict}
        teams={teams}
      />

      {showResults && (
        <PredictionResults
          predictions={predictions}
          homeTeam={homeTeam}
          awayTeam={awayTeam}
          modelPerformanceData={modelPerformanceData}
        />
      )}
    </MainLayout>
  );
};

export default Index;
