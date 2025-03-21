
import React from "react";
import { Player } from "@/types";
import { Card, CardContent } from "@/components/ui/card";
import { Separator } from "@/components/ui/separator";
import { motion } from "framer-motion";

interface TeamPlayersProps {
  teamName: string;
  players: Player[];
  className?: string;
}

const TeamPlayers: React.FC<TeamPlayersProps> = ({ teamName, players, className = "" }) => {
  if (!players || players.length === 0) {
    return (
      <div className={`text-center p-4 ${className}`}>
        <p className="text-muted-foreground">No player data available for {teamName}</p>
      </div>
    );
  }

  const container = {
    hidden: { opacity: 0 },
    show: {
      opacity: 1,
      transition: {
        staggerChildren: 0.1
      }
    }
  };

  const item = {
    hidden: { opacity: 0, y: 20 },
    show: { opacity: 1, y: 0 }
  };

  return (
    <div className={`space-y-3 ${className}`}>
      <h3 className="text-lg font-semibold">{teamName} Players</h3>
      
      <motion.div 
        className="grid grid-cols-1 gap-2"
        variants={container}
        initial="hidden"
        animate="show"
      >
        {players.map((player) => (
          <motion.div key={player.id} variants={item}>
            <Card className="overflow-hidden hover:shadow-md transition-all duration-300">
              <CardContent className="p-3">
                <div className="flex items-center justify-between">
                  <div className="flex items-center space-x-3">
                    <div className="flex-shrink-0 bg-gray-100 rounded-full w-10 h-10 flex items-center justify-center">
                      <span className="font-medium text-sm">{player.position}</span>
                    </div>
                    <div>
                      <h4 className="font-medium">{player.name}</h4>
                      <p className="text-xs text-muted-foreground">{player.position}</p>
                    </div>
                  </div>
                  <div className="flex items-center space-x-2">
                    <span className="text-xs text-muted-foreground">Rating</span>
                    <span className={`px-2 py-1 rounded-full text-xs font-medium ${getRatingColorClass(player.rating)}`}>
                      {player.rating.toFixed(1)}
                    </span>
                  </div>
                </div>
              </CardContent>
            </Card>
          </motion.div>
        ))}
      </motion.div>
    </div>
  );
};

// Helper to get color class based on rating
const getRatingColorClass = (rating: number): string => {
  if (rating >= 9.0) return "bg-green-100 text-green-800";
  if (rating >= 8.5) return "bg-emerald-100 text-emerald-800";
  if (rating >= 8.0) return "bg-teal-100 text-teal-800";
  if (rating >= 7.5) return "bg-blue-100 text-blue-800";
  return "bg-gray-100 text-gray-800";
};

export default TeamPlayers;
