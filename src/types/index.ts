
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

export type TrainingData = {
  homeGoals: number;
  awayGoals: number;
  homeShots: number;
  awayShots: number;
  homeShotsOnTarget: number;
  awayShotsOnTarget: number;
  homeRedCards: number;
  awayRedCards: number;
  result: "Home Win" | "Away Win" | "Draw";
};

export type TrainedModel = {
  name: string;
  predict: (input: number[]) => Promise<{
    outcome: "Home Win" | "Away Win" | "Draw";
    confidence: number;
  }>;
  accuracy: number;
};
