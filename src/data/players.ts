
import { Player } from "@/types";

// Mock player data for teams
export const teamPlayers: Record<string, Player[]> = {
  "Arsenal": [
    { id: "ars1", name: "Bukayo Saka", position: "RW", rating: 8.5 },
    { id: "ars2", name: "Martin Ødegaard", position: "CAM", rating: 8.7 },
    { id: "ars3", name: "Gabriel Jesus", position: "ST", rating: 7.9 },
    { id: "ars4", name: "Declan Rice", position: "CDM", rating: 8.3 },
    { id: "ars5", name: "William Saliba", position: "CB", rating: 8.4 },
  ],
  "Manchester City": [
    { id: "mc1", name: "Erling Haaland", position: "ST", rating: 9.2 },
    { id: "mc2", name: "Kevin De Bruyne", position: "CAM", rating: 9.0 },
    { id: "mc3", name: "Phil Foden", position: "LW", rating: 8.7 },
    { id: "mc4", name: "Rodri", position: "CDM", rating: 8.8 },
    { id: "mc5", name: "Ruben Dias", position: "CB", rating: 8.5 },
  ],
  "Liverpool": [
    { id: "liv1", name: "Mohamed Salah", position: "RW", rating: 8.9 },
    { id: "liv2", name: "Virgil van Dijk", position: "CB", rating: 8.6 },
    { id: "liv3", name: "Alisson", position: "GK", rating: 8.8 },
    { id: "liv4", name: "Trent Alexander-Arnold", position: "RB", rating: 8.5 },
    { id: "liv5", name: "Luis Diaz", position: "LW", rating: 8.2 },
  ],
  "Manchester United": [
    { id: "mu1", name: "Bruno Fernandes", position: "CAM", rating: 8.4 },
    { id: "mu2", name: "Marcus Rashford", position: "LW", rating: 8.2 },
    { id: "mu3", name: "Casemiro", position: "CDM", rating: 8.1 },
    { id: "mu4", name: "Lisandro Martinez", position: "CB", rating: 8.0 },
    { id: "mu5", name: "Rasmus Højlund", position: "ST", rating: 7.8 },
  ],
  "Chelsea": [
    { id: "che1", name: "Cole Palmer", position: "CAM", rating: 8.5 },
    { id: "che2", name: "Nicolas Jackson", position: "ST", rating: 7.8 },
    { id: "che3", name: "Enzo Fernandez", position: "CM", rating: 8.2 },
    { id: "che4", name: "Reece James", position: "RB", rating: 8.4 },
    { id: "che5", name: "Mykhailo Mudryk", position: "LW", rating: 7.7 },
  ],
  "Tottenham Hotspur": [
    { id: "tot1", name: "Son Heung-min", position: "LW", rating: 8.6 },
    { id: "tot2", name: "James Maddison", position: "CAM", rating: 8.1 },
    { id: "tot3", name: "Dejan Kulusevski", position: "RW", rating: 8.0 },
    { id: "tot4", name: "Cristian Romero", position: "CB", rating: 8.2 },
    { id: "tot5", name: "Guglielmo Vicario", position: "GK", rating: 8.0 },
  ],
  "Barcelona": [
    { id: "bar1", name: "Robert Lewandowski", position: "ST", rating: 8.7 },
    { id: "bar2", name: "Lamine Yamal", position: "RW", rating: 8.4 },
    { id: "bar3", name: "Pedri", position: "CM", rating: 8.5 },
    { id: "bar4", name: "Frenkie de Jong", position: "CM", rating: 8.3 },
    { id: "bar5", name: "Ronald Araujo", position: "CB", rating: 8.2 },
  ],
  "Real Madrid": [
    { id: "rm1", name: "Jude Bellingham", position: "CAM", rating: 9.0 },
    { id: "rm2", name: "Vinicius Jr", position: "LW", rating: 8.9 },
    { id: "rm3", name: "Kylian Mbappé", position: "ST", rating: 9.1 },
    { id: "rm4", name: "Antonio Rüdiger", position: "CB", rating: 8.4 },
    { id: "rm5", name: "Thibaut Courtois", position: "GK", rating: 8.8 },
  ],
  "Bayern Munich": [
    { id: "bay1", name: "Harry Kane", position: "ST", rating: 9.0 },
    { id: "bay2", name: "Jamal Musiala", position: "CAM", rating: 8.8 },
    { id: "bay3", name: "Leroy Sané", position: "RW", rating: 8.4 },
    { id: "bay4", name: "Joshua Kimmich", position: "CM", rating: 8.6 },
    { id: "bay5", name: "Manuel Neuer", position: "GK", rating: 8.5 },
  ],
  "Paris Saint-Germain": [
    { id: "psg1", name: "Ousmane Dembélé", position: "RW", rating: 8.3 },
    { id: "psg2", name: "Gonçalo Ramos", position: "ST", rating: 8.1 },
    { id: "psg3", name: "Achraf Hakimi", position: "RB", rating: 8.4 },
    { id: "psg4", name: "Vitinha", position: "CM", rating: 8.2 },
    { id: "psg5", name: "Marquinhos", position: "CB", rating: 8.5 },
  ],
  // Add basic data for other teams to avoid undefined errors
  "Aston Villa": [
    { id: "av1", name: "Ollie Watkins", position: "ST", rating: 8.3 },
    { id: "av2", name: "John McGinn", position: "CM", rating: 8.0 },
    { id: "av3", name: "Emiliano Martinez", position: "GK", rating: 8.5 },
    { id: "av4", name: "Leon Bailey", position: "RW", rating: 8.1 },
    { id: "av5", name: "Tyrone Mings", position: "CB", rating: 7.8 },
  ],
  "Boca Juniors": [
    { id: "bj1", name: "Edinson Cavani", position: "ST", rating: 8.1 },
    { id: "bj2", name: "Cristian Medina", position: "CM", rating: 7.8 },
    { id: "bj3", name: "Marcos Rojo", position: "CB", rating: 7.9 },
    { id: "bj4", name: "Sergio Romero", position: "GK", rating: 7.7 },
    { id: "bj5", name: "Luis Advíncula", position: "RB", rating: 7.5 },
  ],
  "Borussia Dortmund": [
    { id: "bd1", name: "Niclas Füllkrug", position: "ST", rating: 8.1 },
    { id: "bd2", name: "Julian Brandt", position: "CAM", rating: 8.2 },
    { id: "bd3", name: "Mats Hummels", position: "CB", rating: 8.0 },
    { id: "bd4", name: "Gregor Kobel", position: "GK", rating: 8.3 },
    { id: "bd5", name: "Marco Reus", position: "CAM", rating: 8.0 },
  ],
  "Inter Milan": [
    { id: "im1", name: "Lautaro Martínez", position: "ST", rating: 8.7 },
    { id: "im2", name: "Nicolò Barella", position: "CM", rating: 8.5 },
    { id: "im3", name: "Hakan Çalhanoğlu", position: "CDM", rating: 8.4 },
    { id: "im4", name: "Alessandro Bastoni", position: "CB", rating: 8.3 },
    { id: "im5", name: "Yann Sommer", position: "GK", rating: 8.1 },
  ],
  "Juventus": [
    { id: "juv1", name: "Dušan Vlahović", position: "ST", rating: 8.2 },
    { id: "juv2", name: "Federico Chiesa", position: "RW", rating: 8.3 },
    { id: "juv3", name: "Adrien Rabiot", position: "CM", rating: 8.1 },
    { id: "juv4", name: "Bremer", position: "CB", rating: 8.2 },
    { id: "juv5", name: "Wojciech Szczęsny", position: "GK", rating: 8.0 },
  ],
  "Milan": [
    { id: "mil1", name: "Rafael Leão", position: "LW", rating: 8.5 },
    { id: "mil2", name: "Christian Pulisic", position: "RW", rating: 8.2 },
    { id: "mil3", name: "Olivier Giroud", position: "ST", rating: 8.0 },
    { id: "mil4", name: "Theo Hernandez", position: "LB", rating: 8.4 },
    { id: "mil5", name: "Mike Maignan", position: "GK", rating: 8.3 },
  ],
  "River Plate": [
    { id: "rp1", name: "Miguel Borja", position: "ST", rating: 7.9 },
    { id: "rp2", name: "Esequiel Barco", position: "CAM", rating: 7.8 },
    { id: "rp3", name: "Franco Armani", position: "GK", rating: 8.0 },
    { id: "rp4", name: "Paulo Díaz", position: "CB", rating: 7.7 },
    { id: "rp5", name: "Matías Kranevitter", position: "CDM", rating: 7.6 },
  ],
  "West Ham United": [
    { id: "wh1", name: "Mohammed Kudus", position: "RW", rating: 8.1 },
    { id: "wh2", name: "Lucas Paquetá", position: "CAM", rating: 8.0 },
    { id: "wh3", name: "Jarrod Bowen", position: "RW", rating: 8.2 },
    { id: "wh4", name: "Kurt Zouma", position: "CB", rating: 7.9 },
    { id: "wh5", name: "Alphonse Areola", position: "GK", rating: 7.8 },
  ],
};

// Function to get players for a team
export const getTeamPlayers = (teamName: string): Player[] => {
  return teamPlayers[teamName] || [];
};
