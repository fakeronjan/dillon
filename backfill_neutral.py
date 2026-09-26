"""One-off: flag neutral-site regular-season games (London, Mexico City,
Germany, Brazil, ... 2003+) in loaded_NFL_games.csv.

The pre-2026 rows came from Pro Football Reference, which only marked the
Super Bowl as neutral, so every international game was stored as a normal
home game and the ratings engine applied home-field adjustment to it.
nflverse's games.csv has the real location. Match each nflverse neutral
REG game by season + week + final score + team nicknames and set PFR's
marker column to 'N'. Idempotent. Rebuild ratings from scratch afterwards.
"""
import pandas as pd

NFLVERSE = 'https://raw.githubusercontent.com/nflverse/nfldata/master/data/games.csv'
NICK = {'OAK': ['Raiders'], 'LV': ['Raiders'], 'SD': ['Chargers'], 'LAC': ['Chargers'],
        'STL': ['Rams'], 'LA': ['Rams'], 'WAS': ['Redskins', 'Team', 'Commanders']}

n = pd.read_csv(NFLVERSE)
neu = n[(n['location'] == 'Neutral') & (n['game_type'] == 'REG') & n['home_score'].notna()]
L = pd.read_csv('loaded_NFL_games.csv', low_memory=False)
from dillon import NFLVERSE_TEAM_NAMES
def nicks(code):
    return NICK.get(code) or [NFLVERSE_TEAM_NAMES[code].split()[-1]]
hits, misses = 0, []
for g in neu.itertuples():
    wk = L['Week'].astype(str) == str(int(g.week))
    cand = L[(L['Season'] == g.season) & wk]
    hi, lo = max(g.home_score, g.away_score), min(g.home_score, g.away_score)
    cand = cand[(pd.to_numeric(cand['PtsW']) == hi) & (pd.to_numeric(cand['PtsL']) == lo)]
    names = cand['Winner/tie'] + ' | ' + cand['Loser/tie']
    ok = names.apply(lambda s: any(k in s for k in nicks(g.home_team)) and any(k in s for k in nicks(g.away_team)))
    idx = cand[ok].index
    if len(idx) == 0:
        misses.append((g.season, g.week, g.home_team, g.away_team)); continue
    L.loc[idx, 'Unnamed: 5'] = 'N'; hits += len(idx)
L.to_csv('loaded_NFL_games.csv', index=False)
print(f'{len(neu)} nflverse neutral REG games; flagged {hits} rows; unmatched: {misses}')
