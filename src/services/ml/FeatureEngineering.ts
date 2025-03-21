
/**
 * Feature engineering utilities for the ML service.
 * These can be incorporated into the ML service to improve model performance.
 */

// Derive additional features from raw match statistics
export function engineerFeatures(features: number[]): number[] {
  const [
    homeGoals, awayGoals,
    homeShots, awayShots,
    homeShotsOnTarget, awayShotsOnTarget,
    homeRedCards, awayRedCards
  ] = features;
  
  // Create derived features
  const goalDifference = homeGoals - awayGoals;
  const shotDifference = homeShots - awayShots;
  const shotsOnTargetDifference = homeShotsOnTarget - awayShotsOnTarget;
  const redCardDifference = homeRedCards - awayRedCards;
  
  // Shot efficiency (shots on target / total shots)
  const homeShotEfficiency = homeShots > 0 ? homeShotsOnTarget / homeShots : 0;
  const awayShotEfficiency = awayShots > 0 ? awayShotsOnTarget / awayShots : 0;
  const shotEfficiencyDifference = homeShotEfficiency - awayShotEfficiency;
  
  // Scoring efficiency (goals / shots on target)
  const homeScoringEfficiency = homeShotsOnTarget > 0 ? homeGoals / homeShotsOnTarget : 0;
  const awayScoringEfficiency = awayShotsOnTarget > 0 ? awayGoals / awayShotsOnTarget : 0;
  const scoringEfficiencyDifference = homeScoringEfficiency - awayScoringEfficiency;
  
  // Return original features plus derived features
  return [
    ...features,
    goalDifference,
    shotDifference,
    shotsOnTargetDifference,
    redCardDifference,
    homeShotEfficiency,
    awayShotEfficiency,
    shotEfficiencyDifference,
    homeScoringEfficiency,
    awayScoringEfficiency,
    scoringEfficiencyDifference
  ];
}

// Normalize features to have zero mean and unit variance
export function normalizeFeatures(features: number[], means: number[], stds: number[]): number[] {
  return features.map((feature, i) => {
    // Apply z-score normalization, with a small epsilon to avoid division by zero
    return (feature - means[i]) / (stds[i] + 1e-10);
  });
}

// Calculate means and standard deviations for normalization
export function calculateNormalizationParams(data: number[][]): { means: number[], stds: number[] } {
  const numFeatures = data[0].length;
  const means: number[] = Array(numFeatures).fill(0);
  const stds: number[] = Array(numFeatures).fill(0);
  
  // Calculate means
  for (const sample of data) {
    for (let i = 0; i < numFeatures; i++) {
      means[i] += sample[i] / data.length;
    }
  }
  
  // Calculate standard deviations
  for (const sample of data) {
    for (let i = 0; i < numFeatures; i++) {
      stds[i] += Math.pow(sample[i] - means[i], 2) / data.length;
    }
  }
  
  // Take square root to get standard deviations
  for (let i = 0; i < numFeatures; i++) {
    stds[i] = Math.sqrt(stds[i]);
  }
  
  return { means, stds };
}
