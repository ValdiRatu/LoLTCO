### set up path so we can import
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service as ChromeService 
from webdriver_manager.chrome import ChromeDriverManager
import pickle
import time

# setting up webscaper
options = webdriver.ChromeOptions()
options.add_argument("--headless")
options.add_experimental_option('excludeSwitches', ['enable-logging'])
driver = webdriver.Chrome(service = ChromeService(ChromeDriverManager().install()), options=options)

### get summoner name for top players
# looking at top 1k players
max_page = 5 
base_url = "https://u.gg/leaderboards/ranking?region="
regions = ["na1", "euw1", "eun1"]
num_players_per_region = 100*max_page
num_total_players = num_players_per_region*len(regions)

players = []
print("Getting players...")
for j, region in enumerate(regions):
  for i in range(max_page):
    url = base_url + f"{region}" + f"&page={i+1}"
    driver.get(url)
    time.sleep(5) # add time because of slow loading sometimes which causes errors
    elements = driver.find_elements(By.CLASS_NAME, "summoner-name")
    for element in elements:
      players.append(element.text) if element.text != "" else None
    
    expected_amount = (i+1)*100+(j*num_players_per_region)
    if (expected_amount != len(players)):
      raise Exception(f"Error: number of players {len(players)} found does not match expected number {expected_amount}")
    print(f"found {len(players)}/{num_total_players} players")
  
print(f"finished finding top {len(players)} players")

driver.quit()

with open("./data/top_player_summoner_names.pkl", "wb") as file:
  pickle.dump(players, file)
