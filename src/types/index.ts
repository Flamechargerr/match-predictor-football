
export type Team = {
  name: string;
  goals: string;
  shots: string;
  shotsOnTarget: string;
  redCards: string;
};

export type MatchPrediction = {
  outcome: "Home Win" | "Away Win" | "Draw";
  confidence: number;
  modelName: string;
  modelAccuracy: number;
};

export type ModelPerformance = {
  name: string;
  accuracy: number;
  precision: number;
};

export type Player = {
  id: string;
  name: string;
  position: string;
  rating: number;
  image?: string;
};

export type TeamWithPlayers = Team & {
  players: Player[];
};
