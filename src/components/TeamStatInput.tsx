
import React from "react";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Select, SelectContent, SelectGroup, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";

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
  const bgColor = isHome ? "bg-blue-50" : "bg-red-50";
  const borderColor = isHome ? "border-home-light" : "border-away-light";
  const textColor = isHome ? "text-home-dark" : "text-away-dark";
  const labelColor = isHome ? "text-home-DEFAULT" : "text-away-DEFAULT";

  return (
    <div className={`rounded-xl ${bgColor} border ${borderColor} p-5 transition-all duration-300 ${className}`}>
      <h3 className={`text-lg font-semibold mb-4 ${textColor}`}>
        {isHome ? "Home Team" : "Away Team"}
      </h3>

      <div className="space-y-4">
        <div>
          <Label htmlFor={`${teamType}-team`} className={`mb-1.5 block ${labelColor}`}>
            Select {isHome ? "Home" : "Away"} Team *
          </Label>
          <Select value={teamName} onValueChange={onTeamChange}>
            <SelectTrigger id={`${teamType}-team`} className="w-full focused-input bg-white/70">
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
            <Label htmlFor={`${teamType}-goals`} className={`mb-1.5 block ${labelColor}`}>
              Goals *
            </Label>
            <Input
              id={`${teamType}-goals`}
              type="number"
              min="0"
              value={goals}
              onChange={(e) => onGoalsChange(e.target.value)}
              className="focused-input bg-white/70"
              placeholder="0"
            />
          </div>
          <div>
            <Label htmlFor={`${teamType}-shots`} className={`mb-1.5 block ${labelColor}`}>
              Shots *
            </Label>
            <Input
              id={`${teamType}-shots`}
              type="number"
              min="0"
              value={shots}
              onChange={(e) => onShotsChange(e.target.value)}
              className="focused-input bg-white/70"
              placeholder="0"
            />
          </div>
          <div>
            <Label htmlFor={`${teamType}-on-target`} className={`mb-1.5 block ${labelColor}`}>
              On Target *
            </Label>
            <Input
              id={`${teamType}-on-target`}
              type="number"
              min="0"
              value={shotsOnTarget}
              onChange={(e) => onShotsOnTargetChange(e.target.value)}
              className="focused-input bg-white/70"
              placeholder="0"
            />
          </div>
          <div>
            <Label htmlFor={`${teamType}-red-cards`} className={`mb-1.5 block ${labelColor}`}>
              Red Cards *
            </Label>
            <Input
              id={`${teamType}-red-cards`}
              type="number"
              min="0"
              value={redCards}
              onChange={(e) => onRedCardsChange(e.target.value)}
              className="focused-input bg-white/70"
              placeholder="0"
            />
          </div>
        </div>
      </div>
    </div>
  );
};

export default TeamStatInput;
