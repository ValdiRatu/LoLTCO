### set up path so we can import
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pickle
import cassiopeia as Cas

from utils.APIKeyArgExtractor import getAPIKeyFromArgs
from utils.RiotAPIRetryWrapper import RiotAPIRetryWrapper

### get match data from match ids
print("loading match ids...")
with open("../../data/match_ids.pkl", "rb") as file:
  match_ids = pickle.load(file)

print("getting match data...")
key = getAPIKeyFromArgs()
api = RiotAPIRetryWrapper(key)

match_data = []
wrong_version = 0
for match_id in match_ids:
  try:
    platform, id = match_id.split("_")
    if (platform == "NA1"):
      region = Cas.data.Region.north_america
    elif (platform == "EUW1"):
      region = Cas.data.Region.europe_west
    else:
      region = Cas.data.Region.europe_north_east
    match = api.getMatchDataFromMatchID(int(id), region=region)
    main_version, secondary_version = [match["version"].split(".")[i] for i in (0,1)] 
    if (main_version != "13" or secondary_version != "5"):
      print("match not in patch 13.5, skipping...")
      wrong_version += 1
      continue
    match_data.append(api.getMatchDataFromMatchID(int(id), region=region))
    print(f"number of matches found: {len(match_data)}/{len(match_ids)}")
  except Exception as e:
    print(e)
    continue

print(f"total matches found: {len(match_data)}")
print(f"matches not in patch 13.5: {wrong_version}")

with open("../../data/match_data.pkl", "wb") as file:
  pickle.dump(match_data, file)

