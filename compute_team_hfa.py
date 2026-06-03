"""
compute_team_hfa.py — diagnostic for team-specific home field advantage.

For each team T and each window W, computes:
  h_T = mean over T's home games in window W of:
        (home_pts - away_pts) - (T_rating - opp_rating)

Where T_rating and opp_rating are the team ratings GOING INTO the game
(prior-snapshot rating, not post-game). Neutral-site games (Super Bowl,
international) are excluded since neither team has home advantage.

Interpretation: h_T is the team's home margin premium over what its rating
gap alone would predict. League-wide average should land ~2.5 (the global
HCA constant); teams with louder/colder/altitude home environments tend to
sit above; soft-market teams tend to sit below.
"""

import pandas as pd
import numpy as np

LATEST_SEASON = 2025  # most recent completed NFL season (Feb 2026 SB)

games = pd.read_csv('all_NFL_games.csv')
ratings = pd.read_csv('dillon_ratings_with_standings.csv')

# Exclude neutral-site games (SB always; some international games).
games = games[games['is_neutral'] == 0].copy()

# Build prior-snapshot rating per (team, season, week). The rating sitting on
# the row for snapshot (season, week) reflects games THROUGH that week, so we
# shift by one within each team's series to get the rating GOING INTO that week.
ratings = ratings.sort_values(['name', 'ranking_id']).copy()
ratings['rating_prior'] = ratings.groupby('name')['rating'].shift(1)
rating_lookup = ratings.set_index(['name', 'season', 'week'])['rating_prior'].to_dict()


def lookup_rating(name, season, week):
    return rating_lookup.get((name, int(season), int(week)), np.nan)


games['home_rating'] = games.apply(
    lambda g: lookup_rating(g['home_team_name'], g['season'], g['week']), axis=1
)
games['away_rating'] = games.apply(
    lambda g: lookup_rating(g['visitor_team_name'], g['season'], g['week']), axis=1
)

# Drop games where ratings aren't available (early seasons before warmup).
games = games.dropna(subset=['home_rating', 'away_rating']).copy()

# Per-game HFA contribution from the home team's perspective.
games['actual_margin']     = games['home_pts'] - games['visitor_pts']
games['neutral_expected']  = games['home_rating'] - games['away_rating']
games['hfa_contribution']  = games['actual_margin'] - games['neutral_expected']

WINDOWS = {
    '1yr': [LATEST_SEASON],
    '3yr': list(range(LATEST_SEASON - 2, LATEST_SEASON + 1)),
    '5yr': list(range(LATEST_SEASON - 4, LATEST_SEASON + 1)),
}


def team_hfa_table(seasons):
    sub = games[games['season'].isin(seasons)]
    grp = sub.groupby('home_team_name').agg(
        n_home=('hfa_contribution', 'size'),
        hfa=('hfa_contribution', 'mean'),
        std=('hfa_contribution', 'std'),
    ).reset_index()
    grp.columns = ['team', 'n_home', 'hfa', 'std']
    grp['se'] = grp['std'] / np.sqrt(grp['n_home'])
    return grp.sort_values('hfa', ascending=False).reset_index(drop=True)


def league_avg(seasons):
    sub = games[games['season'].isin(seasons)]
    return sub['hfa_contribution'].mean(), len(sub)


# Side-by-side: each team's HFA in all three windows, sorted by 5yr.
tables = {name: team_hfa_table(seasons).set_index('team')
          for name, seasons in WINDOWS.items()}
ranks = {name: tables[name]['hfa'].rank(ascending=False, method='min').astype(int)
         for name in WINDOWS}

# League averages by window — useful baseline reference.
print()
print("League averages (mean HFA across all non-neutral games in window):")
for window_name, seasons in WINDOWS.items():
    league_h, n_games = league_avg(seasons)
    print(f"  {window_name}: {league_h:+.2f} pts/game ({n_games:,} games, seasons {seasons[0]}-{seasons[-1]})")
print(f"  global HCA constant in DILLON.py: +2.50")

print(f"\n{'='*92}")
print(f"  Team HFA across all three windows (sorted by 5yr value, high→low)")
print('=' * 92)
print(f"\n  {'team':<28}  {'1yr HFA (rk)':>14}  {'3yr HFA (rk)':>14}  {'5yr HFA (rk)':>14}")
print(f"  {'-'*28}  {'-'*14}  {'-'*14}  {'-'*14}")

teams_sorted = tables['5yr'].sort_values('hfa', ascending=False).index.tolist()
for team in teams_sorted:
    cells = []
    for name in ['1yr', '3yr', '5yr']:
        if team in tables[name].index:
            v  = tables[name].loc[team, 'hfa']
            rk = ranks[name].loc[team]
            cells.append(f"{v:>+6.2f} (#{rk:>2})")
        else:
            cells.append('       -      ')
    print(f"  {team:<28}  {cells[0]:>14}  {cells[1]:>14}  {cells[2]:>14}")
