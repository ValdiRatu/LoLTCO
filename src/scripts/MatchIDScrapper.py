### set up path so we can import
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pickle
import cassiopeia as Cass

### get match ids from puuids
# open puuid file
print("loading puuids...")
try:
  with open("../../data/player_puuids.pkl", "rb") as file:
    players = pickle.load(file)
except Exception as e:
  print("Failed to load puuids - did you run PlayerScrapper.py?")
  exit()

# get match ids
from utils.RiotAPIRetryWrapper import RiotAPIRetryWrapper
from utils.APIKeyArgExtractor import getAPIKeyFromArgs
import numpy as np 

print("finding most recent matches...")
key = getAPIKeyFromArgs()
api = RiotAPIRetryWrapper(key)

start_time = 1678233600  # March 8, 2023, beginning of patch 13.5
end_time = 1679443200 # March 22, 2023, end of patch 13.5

match_ids = []
for player in players:
  try:
    puuid, region = player["puuid"], player["region"]
    if region == Cass.data.Region.north_america:
      continent = Cass.data.Continent.americas
    else:
      continent = Cass.data.Continent.europe
    
    player_matches = api.getMatchIDsFromPUUID(
      puuid, 
      continent=continent, 
      count=100,
      start = start_time,
      end = end_time
    )
    match_ids.extend(player_matches)
    print(f"found: {len(player_matches)} matches for player {puuid}")
  except Exception as e:
    print(e)
    continue

print(f"total matches found: {len(match_ids)}")
match_ids = np.unique(match_ids)
print(f"found {len(match_ids)} unique matches")
with open("../../data/match_ids_2.pkl", "wb") as file:
  pickle.dump(match_ids, file)