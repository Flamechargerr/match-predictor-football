
import React, { useMemo } from "react";
import { type Player } from "@/types";

interface TeamPlayersProps {
  teamName: string;
  players: Player[];
  showAll: boolean;
  className?: string;
}

const TeamPlayers: React.FC<TeamPlayersProps> = ({
  teamName,
  players,
  showAll,
  className,
}) => {
  const displayedPlayers = useMemo(() => {
    // Show 11 players in advanced view, only 5 in simple view
    return showAll ? players.slice(0, 11) : players.slice(0, 5);
  }, [players, showAll]);

  if (!players.length) {
    return null;
  }

  return (
    <div className={className}>
      <h3 className="text-lg font-semibold mb-2">
        {teamName} {showAll ? "Starting XI" : "Key Players"}
      </h3>
      <ul className="space-y-1">
        {displayedPlayers.map((player, index) => (
          <li key={index} className="flex items-center space-x-2">
            <div className="w-6 h-6 flex items-center justify-center bg-gray-200 text-gray-800 rounded-full text-xs font-semibold">
              {index + 1}
            </div>
            <span className="text-sm font-medium">{player.name}</span>
            <span className="text-xs text-gray-500">{player.position}</span>
          </li>
        ))}
      </ul>
      {!showAll && players.length > 5 && (
        <div className="text-xs text-right italic mt-2 text-gray-500">
          + {players.length - 5} more players
        </div>
      )}
    </div>
  );
};

export default TeamPlayers;
