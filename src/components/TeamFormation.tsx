
import React from "react";
import { Player } from "@/types";
import { Card } from "@/components/ui/card";
import { Trophy, Star, Award } from "lucide-react";

interface TeamFormationProps {
  teamName: string;
  players: Player[];
  formation: string;
  fifaRanking?: number;
  className?: string;
}

const TeamFormation: React.FC<TeamFormationProps> = ({
  teamName,
  players,
  formation,
  fifaRanking,
  className = ""
}) => {
  // Map positions to formation grid areas
  const getPositionStyle = (position: string) => {
    const posMap: Record<string, string> = {
      "GK": "grid-area: gk;",
      "LB": "grid-area: lb;",
      "CB": "grid-area: lcb;",
      "RCB": "grid-area: rcb;",
      "RB": "grid-area: rb;",
      "LDM": "grid-area: ldm;",
      "CDM": "grid-area: cdm;",
      "RDM": "grid-area: rdm;",
      "LM": "grid-area: lm;",
      "CM": "grid-area: lcm;",
      "RCM": "grid-area: rcm;",
      "RM": "grid-area: rm;",
      "LW": "grid-area: lw;",
      "ST": "grid-area: st;",
      "RW": "grid-area: rw;",
      "CAM": "grid-area: cam;",
      "CF": "grid-area: cf;"
    };
    
    return posMap[position] || "";
  };

  // Get the grid template based on formation
  const getFormationGrid = (formation: string) => {
    const formationMap: Record<string, string> = {
      "4-3-3": `
        ".... .... gk  .... ...."
        ".... lb   .... rb   ...."
        ".... lcb  .... rcb  ...."
        "lw   .... cm  .... rw  "
        ".... .... st  .... ...."
      `,
      "4-4-2": `
        ".... .... gk  .... ...."
        ".... lb   .... rb   ...."
        ".... lcb  .... rcb  ...."
        "lm   lcm  .... rcm  rm  "
        ".... cf   .... st   ...."
      `,
      "4-2-3-1": `
        ".... .... gk  .... ...."
        ".... lb   .... rb   ...."
        ".... lcb  .... rcb  ...."
        ".... ldm  .... rdm  ...."
        "lw   .... cam .... rw  "
        ".... .... st  .... ...."
      `,
    };
    
    return formationMap[formation] || formationMap["4-3-3"];
  };

  const formationGrid = getFormationGrid(formation);
  
  // Colors for player rating badges
  const getRatingColor = (rating: number) => {
    if (rating >= 9) return "bg-gradient-to-r from-yellow-400 to-yellow-200 text-yellow-900";
    if (rating >= 8) return "bg-gradient-to-r from-emerald-500 to-emerald-300 text-emerald-900";
    if (rating >= 7) return "bg-gradient-to-r from-blue-500 to-blue-300 text-blue-900";
    return "bg-gradient-to-r from-gray-400 to-gray-200 text-gray-900";
  };

  const teamColorClass = teamName === "Liverpool" || teamName === "Manchester United" || teamName === "Arsenal" ? 
    "from-red-600 to-red-400" : 
    teamName === "Chelsea" || teamName === "Manchester City" ? 
      "from-blue-600 to-blue-400" : 
      "from-purple-600 to-purple-400";

  return (
    <div className={`px-4 py-6 rounded-xl bg-gradient-to-br from-gray-900 to-gray-800 text-white ${className}`}>
      <div className="flex items-center justify-between mb-4">
        <div className="flex items-center">
          <div className={`w-10 h-10 rounded-full bg-gradient-to-r ${teamColorClass} flex items-center justify-center text-white font-bold text-lg shadow-lg`}>
            {teamName.charAt(0)}
          </div>
          <h3 className="ml-3 text-xl font-bold">{teamName}</h3>
        </div>
        <div className="flex items-center">
          <div className="px-3 py-1 rounded-full bg-gray-700 text-sm font-medium">
            {formation}
          </div>
          {fifaRanking && (
            <div className="ml-2 flex items-center px-3 py-1 rounded-full bg-gradient-to-r from-yellow-500 to-amber-500 text-white">
              <Trophy className="w-4 h-4 mr-1" />
              <span className="text-sm font-semibold">#{fifaRanking}</span>
            </div>
          )}
        </div>
      </div>
      
      <div 
        className="relative mt-4 w-full aspect-[4/3] bg-gradient-to-b from-green-800 to-green-900 rounded-lg overflow-hidden p-4"
        style={{
          backgroundImage: "repeating-linear-gradient(to right, rgba(255,255,255,0.1), rgba(255,255,255,0.1) 1px, transparent 1px, transparent 20px), repeating-linear-gradient(to bottom, rgba(255,255,255,0.1), rgba(255,255,255,0.1) 1px, transparent 1px, transparent 20px)",
        }}
      >
        <div className="pitch-lines absolute inset-0">
          {/* Center circle */}
          <div className="absolute left-1/2 top-1/2 w-24 h-24 border-2 border-white/20 rounded-full -translate-x-1/2 -translate-y-1/2"></div>
          {/* Center line */}
          <div className="absolute left-0 top-1/2 w-full h-0.5 bg-white/20"></div>
          {/* Penalty areas */}
          <div className="absolute left-1/2 bottom-0 w-40 h-16 border-t-2 border-x-2 border-white/20 -translate-x-1/2"></div>
          <div className="absolute left-1/2 top-0 w-40 h-16 border-b-2 border-x-2 border-white/20 -translate-x-1/2"></div>
        </div>

        <div 
          className="grid h-full" 
          style={{ 
            gridTemplateColumns: "1fr 1fr 1fr 1fr 1fr",
            gridTemplateRows: "1fr 1fr 1fr 1fr 1fr",
            gridTemplateAreas: formationGrid
          }}
        >
          {players.slice(0, 11).map((player) => (
            <div
              key={player.id}
              className="flex flex-col items-center justify-center"
              style={{ [getPositionStyle(player.position).split(":")[0]]: getPositionStyle(player.position).split(":")[1] }}
            >
              <div className={`w-11 h-11 rounded-full ${getRatingColor(player.rating)} flex items-center justify-center shadow-lg mb-1 text-sm font-bold relative`}>
                {player.position === "GK" ? "GK" : player.name.split(" ")[0].charAt(0) + player.name.split(" ").pop()?.charAt(0)}
                <div className="absolute -bottom-1 -right-1 w-5 h-5 rounded-full bg-black/70 text-white flex items-center justify-center text-[10px]">
                  {player.rating.toFixed(1)}
                </div>
              </div>
              <span className="text-xs font-semibold text-white/80 bg-black/30 px-1 rounded">
                {player.name.split(" ").pop()}
              </span>
            </div>
          ))}
        </div>
      </div>

      <div className="mt-4 grid grid-cols-3 gap-2">
        <div className="bg-gradient-to-r from-indigo-800/50 to-indigo-700/50 rounded-lg p-3 text-center">
          <Star className="w-5 h-5 mx-auto mb-1 text-indigo-300" />
          <div className="text-sm text-indigo-200">Top Player</div>
          <div className="font-bold mt-1">{players.sort((a, b) => b.rating - a.rating)[0]?.name || "N/A"}</div>
        </div>
        <div className="bg-gradient-to-r from-emerald-800/50 to-emerald-700/50 rounded-lg p-3 text-center">
          <Award className="w-5 h-5 mx-auto mb-1 text-emerald-300" />
          <div className="text-sm text-emerald-200">Avg Rating</div>
          <div className="font-bold mt-1">
            {players.length > 0 
              ? (players.reduce((sum, p) => sum + p.rating, 0) / players.length).toFixed(1) 
              : "N/A"}
          </div>
        </div>
        <div className="bg-gradient-to-r from-amber-800/50 to-amber-700/50 rounded-lg p-3 text-center">
          <Trophy className="w-5 h-5 mx-auto mb-1 text-amber-300" />
          <div className="text-sm text-amber-200">League Pos</div>
          <div className="font-bold mt-1">{Math.floor(Math.random() * 10) + 1}</div>
        </div>
      </div>
    </div>
  );
};

export default TeamFormation;
