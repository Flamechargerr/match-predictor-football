
import React from "react";
import { Button } from "@/components/ui/button";
import { toast } from "@/components/ui/use-toast";
import TeamStatInput from "@/components/TeamStatInput";
import PredictIcon from "@/components/PredictIcon";
import { type Team, type Player } from "@/types";
import TeamDisplay from "@/components/TeamDisplay";

interface TeamInputFormProps {
  homeTeam: Team;
  setHomeTeam: (team: Team) => void;
  awayTeam: Team;
  setAwayTeam: (team: Team) => void;
  homePlayers: Player[];
  awayPlayers: Player[];
  showAdvancedView: boolean;
  teamFormations: Record<string, string>;
  teamRankings: Record<string, number>;
  isLoading: boolean;
  onPredict: () => void;
  teams: string[];
}

const TeamInputForm: React.FC<TeamInputFormProps> = ({
  homeTeam,
  setHomeTeam,
  awayTeam,
  setAwayTeam,
  homePlayers,
  awayPlayers,
  showAdvancedView,
  teamFormations,
  teamRankings,
  isLoading,
  onPredict,
  teams,
}) => {
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

  return (
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

      <TeamDisplay 
        homeTeam={homeTeam}
        awayTeam={awayTeam}
        homePlayers={homePlayers}
        awayPlayers={awayPlayers}
        showAdvancedView={showAdvancedView}
        teamFormations={teamFormations}
        teamRankings={teamRankings}
      />

      <Button 
        size="lg" 
        onClick={onPredict} 
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
  );
};

export default TeamInputForm;
