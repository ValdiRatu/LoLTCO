import pickle
import numpy as np
import json
from cassiopeia import data as LeagueData
from sklearn.model_selection import train_test_split

with open("../../tarball_data/champions.json", "rb") as f:
    championData = json.load(f)["data"]
roles = ["TOP", "JUNGLE", "MIDDLE", "BOTTOM", "UTILITY"]

championToIndexMap = {list(championData.keys())[i].lower(): i+1 for i in range(len(championData.keys()))}
roleToIndexMap = {roles[i]: i+1 for i in range(len(roles))}
numChamps = 163
numRoles = len(roles)

def getDataOneHot(match = "match_data_subset.pkl", data = None):
  if data is None:
    with open(f"../../data/{match}", "rb") as f:
      matchData = pickle.load(f)
  else:
    matchData = data
  X = np.zeros((len(matchData), numChamps * 2 * numRoles))
  y = np.zeros((len(matchData), 1))

  for i, match in enumerate(matchData):
    players = np.array([team["participants"] for team in match["teams"]]).flatten()
    for player in players:
      teamIndex = 0 if player["side"] == LeagueData.Side.blue else numChamps*numRoles
      champIndex = championToIndexMap[player["championName"].lower()]-1 # -1 because champions use 1-indexing
      roleIndex = roleToIndexMap[player["teamPosition"]]-1 if player["teamPosition"] != "" else getUnknownPosition(players)  # -1 because roles use 1-indexing
      featureIndex = teamIndex + roleIndex*numChamps+champIndex
      X[i, featureIndex] = 1
    y[i] = did_blue_win(match) 

  return X, y

def getDataEmbeddings(match = "match_data_subset.pkl", data = None):
  if data is None:
    with open(f"../../data/{match}", "rb") as f:
      matchData = pickle.load(f)
  else:
    matchData = data
  X = np.zeros((len(matchData), 20), dtype=int)
  y = np.zeros((len(matchData), 1), dtype=int)

  for i, match in enumerate(matchData):
    teamA, teamB = match["teams"]
    teamBlue, teamRed = [teamA, teamB] if teamA["side"] == LeagueData.Side.blue else [teamB, teamA]

    roleIndex = 10
    for j, player in enumerate(teamBlue["participants"]):
      X[i, j] = championToIndexMap[player["championName"].lower()]
      X[i, j+roleIndex] = roleToIndexMap[player["teamPosition"]] if player["teamPosition"] != "" else getUnknownPosition(teamBlue["participants"])
    
    teamIndex = 5
    for j, player in enumerate(teamRed["participants"]):
      X[i, j+teamIndex] = championToIndexMap[player["championName"].lower()]
      X[i, j+roleIndex+teamIndex] = roleToIndexMap[player["teamPosition"]] if player["teamPosition"] != "" else getUnknownPosition(teamRed["participants"])
    
    y[i] = teamBlue["isWinner"]
  
  return X, y


def did_blue_win(match):
  team_1, team_2 = match["teams"]
  if team_1["side"] == LeagueData.Side.blue:
    return team_1["isWinner"]
  return team_2["isWinner"] 

def getUnknownPosition(players):
  positionCount = np.zeros(numRoles)
  for player in players :
    if player["teamPosition"] != "":
      positionCount[roleToIndexMap[player["teamPosition"]]-1] += 1
  return np.argmin(positionCount)

def getTrainValData():
  with open("../../data/match_data.pkl", "rb") as f:
    matchData = pickle.load(f)
  trainData, valData = train_test_split(matchData, test_size=0.1129, random_state=42)

  return trainData, valData