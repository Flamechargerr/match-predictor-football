
import React from "react";
import { type Player } from "@/types";
import TeamPlayers from "@/components/TeamPlayers";
import TeamFormation from "@/components/TeamFormation";

interface TeamDisplayProps {
  homeTeam: {
    name: string;
  };
  awayTeam: {
    name: string;
  };
  homePlayers: Player[];
  awayPlayers: Player[];
  showAdvancedView: boolean;
  teamFormations: Record<string, string>;
  teamRankings: Record<string, number>;
}

const TeamDisplay: React.FC<TeamDisplayProps> = ({
  homeTeam,
  awayTeam,
  homePlayers,
  awayPlayers,
  showAdvancedView,
  teamFormations,
  teamRankings,
}) => {
  if (!homeTeam.name && !awayTeam.name) {
    return null;
  }

  return (
    <>
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
    </>
  );
};

export default TeamDisplay;
