from datafc.sofascore import (
    match_data,
    match_odds_data,
    match_stats_data,
    momentum_data,
    lineups_data,
    coordinates_data,
    substitutions_data,
    goal_networks_data,
    shots_data,
    standings_data
)
import pandas as pd
import json
import numpy as np
from tqdm import tqdm  # For progress bar

# Create an empty list to store dataframes for each week
all_matches = []

# Fetch data for weeks 1 through 25
for week in range(1, 30):
    week_df = match_data(
        tournament_id=52,
        season_id=63814,
        week_number=week
    )
    all_matches.append(week_df)
    print(f"Fetched data for week {week}")

# Combine all dataframes into a single dataframe
match_df = pd.concat(all_matches, ignore_index=True)
print(f"Total matches fetched: {len(match_df)}")
data=match_df.copy()

data=data[['country','tournament','season','week','home_team','away_team','home_score_display','away_score_display']]
data.rename(columns={
    'home_score_display': 'home_score',
    'away_score_display': 'away_score'
}, inplace=True)

data=data[data['home_score']!=""]
json_data = data.to_json(orient="records", indent=2)

# JSON verisini dosyaya kaydediyoruz.
json_file_path = "match_results.json"
with open(json_file_path, "w") as json_file:
    json_file.write(json_data)