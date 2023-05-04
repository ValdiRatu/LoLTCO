from tenacity import retry, wait_fixed
import cassiopeia as Cass

class RiotAPIRetryWrapper:
  def __init__(self, apiKey):
    print(f"API key: {apiKey}")
    self.Cass = Cass
    self.Cass.apply_settings({"logging": {"print_calls": False, "print_riot_api_key": False}})
    self.Cass.set_riot_api_key(apiKey)

  def GetPUUIDFromSummoner(self, summonerName, region = Cass.data.Region.north_america):
    try:
      summoner = self.Cass.get_summoner(name=summonerName, region=region)
      return {
        "puuid": summoner.puuid,
        "region": summoner.region
      }
    except Exception as e:
      print(e)
      raise Exception(f"Failed to get puuid for {summonerName} in {region}, skipping...")
    
  def getMatchIDsFromPUUID(
      self, 
      puuid, 
      start = None,
      end = None,
      continent = Cass.data.Continent.americas, 
      queue = Cass.data.Queue.ranked_solo_fives,
      type = Cass.data.MatchType.ranked,
      count = 20
    ):
    matches = self.Cass.get_match_history(
      puuid = puuid,
      continent = continent,
      queue = queue,
      type = type,
      count = count,
      start_time = start,
      end_time = end
    )
    try: 
      match_ids = [f"{match.platform.value}_{match.id}" for match in matches] 
      return match_ids
    except Exception as e:
      print(e)
      raise Exception(f"Failed to get match ids for {puuid}, skipping...")
    
  def getMatchDataFromMatchID(self, matchId, region = Cass.data.Region.north_america):
    try:
      match = self.Cass.get_match(id=matchId, region=region)
      match.participants # call this because its lazy loaded
      match = match.to_dict()
      return {
        "matchId": f"{match['platform']}_{match['id']}",
        "teams": match["teams"],
        "version": match["version"],
      }
    except Exception as e:
      print(e)
      raise Exception(f"Failed to get match data for match: {matchId} in region: {region}, skipping...")
