### set up path so we can import
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pickle
import cassiopeia as Cass

from utils.RiotAPIRetryWrapper import RiotAPIRetryWrapper
from utils.APIKeyArgExtractor import getAPIKeyFromArgs

### loading summoner names:
print("loading summoner names...")
with open("./data/top_player_summoner_names.pkl", "rb") as file:
  players = pickle.load(file)

### get puuids from summoner names
print("Getting puuids...")
key = getAPIKeyFromArgs()
api = RiotAPIRetryWrapper(key)
# yeah there is coupling here with the SummonerNameScrapper.py script and the
# number of players per region as well as the order of the regions in the regions list,
# but it's not a big deal. 
num_players_per_region = 100*5

puuids = []
for i, player in enumerate(players):
  try:
    if (i // num_players_per_region < 1):
      region = Cass.data.Region.north_america 
    elif (i // num_players_per_region < 2):
      region = Cass.data.Region.europe_west
    else:
      region = Cass.data.Region.europe_north_east
    puuids.append(api.GetPUUIDFromSummoner(player, region))
    print(f"found {len(puuids)}/{len(players)} puuids")
  except Exception as e:
    print(e)
    continue

with open("./data/player_puuids.pkl", "wb") as file:
  pickle.dump(puuids, file)




