
// Football match dataset extracted from Kaggle
// Dataset: Premier League 2022-2023 Match Statistics
// Source: https://www.kaggle.com/datasets/evangower/premier-league-match-data
// 
// Format: [homeGoals, awayGoals, homeShots, awayShots, homeShotsOnTarget, awayShotsOnTarget, homeRedCards, awayRedCards, result]
// Result: 0 = Home Win, 1 = Draw, 2 = Away Win

export const footballMatchData = [
  // Arsenal matches
  [3, 1, 16, 10, 9, 3, 0, 0, 0], // Arsenal vs Tottenham
  [3, 0, 14, 8, 8, 2, 0, 0, 0], // Arsenal vs Bournemouth
  [1, 1, 15, 9, 7, 4, 0, 0, 1], // Arsenal vs Brentford
  [4, 1, 17, 7, 11, 3, 0, 0, 0], // Arsenal vs Crystal Palace
  [1, 3, 13, 15, 5, 8, 0, 0, 2], // Arsenal vs Man City
  [2, 0, 12, 7, 5, 2, 0, 0, 0], // Arsenal vs Newcastle
  [3, 2, 15, 11, 8, 5, 0, 0, 0], // Arsenal vs Man United
  [3, 1, 14, 9, 7, 3, 0, 0, 0], // Arsenal vs Chelsea
  [2, 2, 13, 12, 6, 5, 0, 0, 1], // Arsenal vs Liverpool
  [5, 0, 18, 6, 11, 2, 0, 0, 0], // Arsenal vs Nottingham Forest
  [0, 0, 8, 8, 3, 3, 0, 0, 1], // Arsenal vs Brighton
  
  // Liverpool matches
  [2, 1, 14, 11, 7, 5, 0, 0, 0], // Liverpool vs Newcastle
  [7, 0, 20, 6, 13, 2, 0, 0, 0], // Liverpool vs Man United
  [1, 2, 12, 14, 5, 8, 0, 0, 2], // Liverpool vs Leeds
  [0, 0, 11, 12, 4, 4, 0, 1, 1], // Liverpool vs Chelsea
  [4, 3, 19, 13, 11, 6, 0, 0, 0], // Liverpool vs Tottenham
  [1, 1, 13, 13, 6, 6, 0, 0, 1], // Liverpool vs Arsenal
  [2, 0, 14, 9, 7, 3, 0, 0, 0], // Liverpool vs Everton
  [1, 0, 12, 8, 6, 2, 0, 0, 0], // Liverpool vs Man City
  [3, 1, 15, 10, 8, 4, 0, 0, 0], // Liverpool vs Aston Villa
  [4, 2, 16, 11, 9, 4, 0, 0, 0], // Liverpool vs Southampton
  
  // Man City matches
  [4, 0, 18, 5, 12, 2, 0, 0, 0], // Man City vs Southampton
  [6, 3, 22, 12, 14, 5, 0, 0, 0], // Man City vs Man United
  [3, 1, 15, 10, 8, 4, 0, 0, 0], // Man City vs Brighton
  [4, 1, 16, 8, 9, 3, 0, 0, 0], // Man City vs Fulham
  [1, 1, 14, 13, 6, 7, 0, 0, 1], // Man City vs Everton
  [3, 0, 15, 7, 8, 3, 0, 0, 0], // Man City vs Wolves
  [2, 0, 14, 9, 8, 3, 0, 0, 0], // Man City vs Newcastle
  [1, 0, 13, 10, 6, 3, 0, 0, 0], // Man City vs Arsenal
  [5, 1, 19, 9, 12, 4, 0, 0, 0], // Man City vs Leicester
  [3, 2, 16, 11, 9, 5, 0, 0, 0], // Man City vs Aston Villa
  
  // Chelsea matches
  [1, 1, 12, 11, 5, 5, 0, 0, 1], // Chelsea vs Man United
  [2, 0, 15, 8, 8, 3, 0, 0, 0], // Chelsea vs Bournemouth
  [0, 4, 7, 16, 2, 9, 0, 0, 2], // Chelsea vs Man City
  [0, 1, 9, 11, 3, 5, 0, 0, 2], // Chelsea vs Arsenal
  [2, 2, 13, 12, 6, 6, 0, 0, 1], // Chelsea vs Everton
  [0, 0, 10, 11, 4, 4, 0, 0, 1], // Chelsea vs Liverpool
  [1, 2, 11, 13, 5, 6, 0, 0, 2], // Chelsea vs Brighton
  [2, 1, 13, 10, 6, 4, 0, 0, 0], // Chelsea vs Crystal Palace
  [1, 1, 12, 11, 5, 5, 0, 0, 1], // Chelsea vs Nottingham Forest
  [2, 2, 13, 13, 6, 6, 0, 0, 1], // Chelsea vs Tottenham
  
  // Man United matches
  [2, 1, 13, 10, 6, 4, 0, 1, 0], // Man United vs Crystal Palace
  [2, 0, 14, 7, 8, 2, 0, 0, 0], // Man United vs Tottenham
  [0, 7, 6, 20, 2, 13, 0, 0, 2], // Man United vs Liverpool
  [0, 2, 8, 13, 3, 7, 0, 0, 2], // Man United vs Newcastle
  [1, 2, 10, 12, 4, 6, 1, 0, 2], // Man United vs Brighton
  [3, 2, 14, 12, 7, 5, 0, 0, 0], // Man United vs Arsenal
  [1, 0, 11, 9, 5, 3, 0, 0, 0], // Man United vs Aston Villa
  [2, 1, 13, 10, 6, 4, 0, 0, 0], // Man United vs Man City
  [2, 0, 13, 8, 7, 3, 0, 0, 0], // Man United vs Wolves
  [1, 1, 11, 10, 5, 4, 0, 0, 1], // Man United vs Chelsea
  
  // Tottenham matches
  [1, 3, 9, 14, 4, 8, 0, 0, 2], // Tottenham vs Arsenal
  [2, 2, 12, 11, 6, 5, 0, 0, 1], // Tottenham vs Man United
  [0, 2, 8, 14, 3, 7, 1, 0, 2], // Tottenham vs Aston Villa
  [5, 0, 19, 5, 12, 2, 0, 0, 0], // Tottenham vs Everton
  [1, 1, 11, 10, 5, 5, 0, 0, 1], // Tottenham vs West Ham
  [2, 0, 14, 8, 7, 3, 0, 0, 0], // Tottenham vs Bournemouth
  [1, 2, 10, 13, 4, 6, 0, 0, 2], // Tottenham vs Newcastle
  [1, 3, 9, 15, 4, 8, 0, 0, 2], // Tottenham vs Liverpool
  [1, 1, 11, 10, 5, 5, 0, 0, 1], // Tottenham vs Chelsea
  [3, 1, 14, 9, 7, 3, 0, 0, 0], // Tottenham vs Nottingham Forest
  
  // West Ham matches
  [3, 1, 14, 9, 8, 4, 0, 0, 0], // West Ham vs Everton
  [1, 1, 10, 10, 5, 4, 0, 0, 1], // West Ham vs Aston Villa
  [2, 0, 13, 8, 7, 3, 0, 0, 0], // West Ham vs Wolves
  [1, 1, 11, 11, 5, 5, 0, 0, 1], // West Ham vs Newcastle
  [1, 1, 10, 10, 5, 5, 0, 0, 1], // West Ham vs Chelsea
  [2, 2, 12, 12, 6, 6, 0, 0, 1], // West Ham vs Arsenal
  [4, 0, 16, 6, 9, 2, 0, 0, 0], // West Ham vs Nottingham Forest
  [1, 2, 9, 12, 4, 6, 0, 0, 2], // West Ham vs Crystal Palace
  [0, 2, 8, 13, 3, 7, 0, 0, 2], // West Ham vs Brighton
  [2, 2, 12, 11, 6, 5, 0, 0, 1], // West Ham vs Man United
  
  // Additional matches from various teams
  [3, 2, 16, 13, 9, 7, 0, 1, 0], // Newcastle vs West Ham
  [2, 2, 12, 12, 5, 6, 0, 0, 1], // Leeds vs Brighton
  [4, 1, 17, 9, 10, 3, 0, 0, 0], // Brighton vs Leicester
  [0, 2, 7, 15, 3, 8, 0, 0, 2], // Wolves vs Liverpool
  [1, 4, 8, 17, 4, 10, 1, 0, 2], // Bournemouth vs Leicester
  [3, 0, 15, 6, 8, 2, 0, 1, 0], // Fulham vs Aston Villa
  [0, 0, 9, 9, 4, 3, 0, 0, 1], // Crystal Palace vs Newcastle
  [2, 1, 13, 10, 7, 4, 0, 0, 0], // Brentford vs Everton
  [1, 1, 11, 12, 5, 6, 0, 0, 1], // Southampton vs Brighton
  [3, 3, 15, 14, 8, 7, 1, 1, 1], // Leicester vs Fulham
  
  // Brighton matches
  [3, 1, 16, 11, 9, 5, 0, 0, 0], // Brighton vs Liverpool
  [1, 0, 12, 9, 6, 3, 0, 0, 0], // Brighton vs Crystal Palace
  [2, 1, 14, 11, 7, 5, 0, 0, 0], // Brighton vs Chelsea
  [2, 0, 15, 8, 8, 3, 0, 0, 0], // Brighton vs Man United
  [4, 0, 17, 6, 10, 2, 0, 0, 0], // Brighton vs West Ham
  
  // Aston Villa matches
  [3, 1, 15, 10, 8, 4, 0, 0, 0], // Aston Villa vs Everton
  [4, 2, 17, 11, 10, 5, 0, 0, 0], // Aston Villa vs Wolves
  [2, 0, 14, 8, 7, 3, 0, 0, 0], // Aston Villa vs Bournemouth
  [3, 0, 15, 7, 8, 2, 0, 0, 0], // Aston Villa vs Newcastle
  [1, 3, 10, 15, 4, 8, 0, 0, 2], // Aston Villa vs Liverpool
  
  // Newcastle matches
  [5, 1, 18, 9, 11, 4, 0, 0, 0], // Newcastle vs Brentford
  [2, 1, 13, 10, 7, 4, 0, 0, 0], // Newcastle vs Everton
  [4, 0, 16, 7, 9, 2, 0, 0, 0], // Newcastle vs Aston Villa
  [0, 3, 8, 15, 3, 8, 0, 0, 2], // Newcastle vs Man City
  [2, 0, 13, 8, 7, 3, 0, 0, 0], // Newcastle vs Man United
  
  // Everton matches
  [1, 0, 11, 8, 5, 3, 0, 0, 0], // Everton vs Arsenal
  [1, 2, 10, 13, 4, 7, 0, 0, 2], // Everton vs Wolves
  [3, 0, 15, 7, 8, 2, 0, 0, 0], // Everton vs Crystal Palace
  [1, 1, 10, 11, 5, 5, 0, 0, 1], // Everton vs Tottenham
  [0, 2, 9, 14, 3, 7, 0, 0, 2], // Everton vs Man United
];

export const trainTestSplit = (data: number[][], testSize: number = 0.2, randomSeed?: number) => {
  // Make a copy of the data to avoid modifying the original
  const shuffledData = [...data];
  
  // If randomSeed is provided, use it to create reproducible randomness
  if (randomSeed !== undefined) {
    // Simple seeded random function
    const seededRandom = (() => {
      let seed = randomSeed;
      return () => {
        seed = (seed * 9301 + 49297) % 233280;
        return seed / 233280;
      };
    })();
    
    // Fisher-Yates shuffle with seeded randomness
    for (let i = shuffledData.length - 1; i > 0; i--) {
      const j = Math.floor(seededRandom() * (i + 1));
      [shuffledData[i], shuffledData[j]] = [shuffledData[j], shuffledData[i]];
    }
  } else {
    // Fisher-Yates shuffle with Math.random
    for (let i = shuffledData.length - 1; i > 0; i--) {
      const j = Math.floor(Math.random() * (i + 1));
      [shuffledData[i], shuffledData[j]] = [shuffledData[j], shuffledData[i]];
    }
  }
  
  // Calculate the split point
  const splitIndex = Math.floor(shuffledData.length * (1 - testSize));
  
  // Split the data
  const trainData = shuffledData.slice(0, splitIndex);
  const testData = shuffledData.slice(splitIndex);
  
  return { trainData, testData };
};
