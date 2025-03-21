
import React from "react";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Select, SelectContent, SelectGroup, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { motion } from "framer-motion";

interface TeamStatInputProps {
  teamType: "home" | "away";
  teamName: string;
  onTeamChange: (team: string) => void;
  goals: string;
  onGoalsChange: (goals: string) => void;
  shots: string;
  onShotsChange: (shots: string) => void;
  shotsOnTarget: string;
  onShotsOnTargetChange: (shotsOnTarget: string) => void;
  redCards: string;
  onRedCardsChange: (redCards: string) => void;
  teamOptions: string[];
  className?: string;
}

const TeamStatInput: React.FC<TeamStatInputProps> = ({
  teamType,
  teamName,
  onTeamChange,
  goals,
  onGoalsChange,
  shots,
  onShotsChange,
  shotsOnTarget,
  onShotsOnTargetChange,
  redCards,
  onRedCardsChange,
  teamOptions,
  className = "",
}) => {
  const isHome = teamType === "home";
  const bgGradient = isHome 
    ? "bg-gradient-to-br from-blue-50 to-blue-50/70" 
    : "bg-gradient-to-br from-red-50 to-red-50/70";
  const borderColor = isHome ? "border-home-light" : "border-away-light";
  const textColor = isHome ? "text-home-dark" : "text-away-dark";
  const labelColor = isHome ? "text-home-DEFAULT" : "text-away-DEFAULT";
  const shadowColor = isHome ? "shadow-blue-100/50" : "shadow-red-100/50";

  return (
    <motion.div 
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.4, ease: "easeOut" }}
      className={`stat-input-card ${bgGradient} border ${borderColor} ${shadowColor} ${className}`}
    >
      <div className="flex items-center mb-4 space-x-2">
        <div className={`w-3 h-3 rounded-full ${isHome ? "bg-home-DEFAULT" : "bg-away-DEFAULT"}`}></div>
        <h3 className={`text-lg font-semibold ${textColor}`}>
          {isHome ? "Home Team" : "Away Team"}
        </h3>
      </div>

      <div className="space-y-4">
        <div>
          <Label htmlFor={`${teamType}-team`} className={`mb-1.5 block font-medium ${labelColor}`}>
            Select {isHome ? "Home" : "Away"} Team *
          </Label>
          <Select value={teamName} onValueChange={onTeamChange}>
            <SelectTrigger id={`${teamType}-team`} className="w-full focused-input bg-white/80 backdrop-blur-sm">
              <SelectValue placeholder={`Select ${isHome ? "Home" : "Away"} Team`} />
            </SelectTrigger>
            <SelectContent>
              <SelectGroup>
                {teamOptions.map((team) => (
                  <SelectItem key={team} value={team}>
                    {team}
                  </SelectItem>
                ))}
              </SelectGroup>
            </SelectContent>
          </Select>
        </div>

        <div className="grid grid-cols-2 gap-4">
          <div>
            <Label htmlFor={`${teamType}-goals`} className={`mb-1.5 block font-medium ${labelColor}`}>
              Goals *
            </Label>
            <Input
              id={`${teamType}-goals`}
              type="number"
              min="0"
              value={goals}
              onChange={(e) => onGoalsChange(e.target.value)}
              className="focused-input bg-white/80 backdrop-blur-sm"
              placeholder="0"
            />
          </div>
          <div>
            <Label htmlFor={`${teamType}-shots`} className={`mb-1.5 block font-medium ${labelColor}`}>
              Shots *
            </Label>
            <Input
              id={`${teamType}-shots`}
              type="number"
              min="0"
              value={shots}
              onChange={(e) => onShotsChange(e.target.value)}
              className="focused-input bg-white/80 backdrop-blur-sm"
              placeholder="0"
            />
          </div>
          <div>
            <Label htmlFor={`${teamType}-on-target`} className={`mb-1.5 block font-medium ${labelColor}`}>
              On Target *
            </Label>
            <Input
              id={`${teamType}-on-target`}
              type="number"
              min="0"
              value={shotsOnTarget}
              onChange={(e) => onShotsOnTargetChange(e.target.value)}
              className="focused-input bg-white/80 backdrop-blur-sm"
              placeholder="0"
            />
          </div>
          <div>
            <Label htmlFor={`${teamType}-red-cards`} className={`mb-1.5 block font-medium ${labelColor}`}>
              Red Cards *
            </Label>
            <Input
              id={`${teamType}-red-cards`}
              type="number"
              min="0"
              value={redCards}
              onChange={(e) => onRedCardsChange(e.target.value)}
              className="focused-input bg-white/80 backdrop-blur-sm"
              placeholder="0"
            />
          </div>
        </div>
      </div>
    </motion.div>
  );
};

export default TeamStatInput;
