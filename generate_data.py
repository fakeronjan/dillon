"""
generate_data.py - reads dillon_ratings_with_standings.csv and writes JSON for the DILLON web frontend.
Run after dillon.py. Outputs to docs/data/.

NFL-specific tweaks vs LOBO/DUNCAN:
  - REACT 20-week rolling rating, decomposed into Offense + Defense via a
    Massey-style dual solver (engine commit 534caff)
  - AFC/NFC conference + division mapping for all 44 historical team names since 1970
  - Weekly cadence (cume_week_id, not daily)
  - Records can include ties (W-L-T), e.g. "10-5-1"
  - Snapshot date derived from the LATEST actual game date in that season-week
"""

import pandas as pd
import numpy as np
import json
import os
import re
import collections
from bisect import bisect_right
from datetime import datetime, timezone

os.makedirs('docs/data/teams', exist_ok=True)
os.makedirs('docs/data/seasons', exist_ok=True)

print("Reading ratings...")
df = pd.read_csv('dillon_ratings_with_standings.csv')

games = pd.read_csv('all_NFL_games.csv')
games['date'] = pd.to_datetime(games['date'], errors='coerce')
# Y2K fix: pandas parses "9/20/70" as 2070-09-20, but it's 1970-09-20.
# Anything past 2030 is wrong (NFL hasn't played those games yet).
_y2k_mask = games['date'].dt.year > 2030
games.loc[_y2k_mask, 'date'] = games.loc[_y2k_mask, 'date'].apply(lambda d: d.replace(year=d.year - 100))

# Build season_week → date (use the LATEST game date in that week)
_sw_to_date = (
    games.dropna(subset=['date'])
    .groupby('season_week')['date']
    .max()
    .dt.date
    .to_dict()
)


def sw_to_date_str(sw):
    d = _sw_to_date.get(float(sw))
    return str(d) if d else ''


df['date'] = df['season_week'].apply(sw_to_date_str)


# ── Conference + Division mapping (covers all 44 team names since 1970) ──────
# Uses CURRENT division of the franchise lineage. Pre-2002 NFL had a different
# 6-division layout, so historical teams are mapped to where their lineage sits today.
TEAM_CONFERENCE = {
    # AFC East
    'Buffalo Bills':            ('AFC', 'East'),
    'Miami Dolphins':           ('AFC', 'East'),
    'New England Patriots':     ('AFC', 'East'),
    'Boston Patriots':          ('AFC', 'East'),  # → New England 1971
    'New York Jets':            ('AFC', 'East'),

    # AFC North (post-2002)
    'Baltimore Ravens':         ('AFC', 'North'),
    'Cincinnati Bengals':       ('AFC', 'North'),
    'Cleveland Browns':         ('AFC', 'North'),
    'Pittsburgh Steelers':      ('AFC', 'North'),

    # AFC South (post-2002, new division)
    'Houston Texans':           ('AFC', 'South'),
    'Indianapolis Colts':       ('AFC', 'South'),
    'Baltimore Colts':          ('AFC', 'South'),  # → Indianapolis Colts 1984
    'Jacksonville Jaguars':     ('AFC', 'South'),
    'Tennessee Titans':         ('AFC', 'South'),
    'Tennessee Oilers':         ('AFC', 'South'),  # → Tennessee Titans 1999
    'Houston Oilers':           ('AFC', 'South'),  # → Tennessee Oilers/Titans

    # AFC West
    'Denver Broncos':           ('AFC', 'West'),
    'Kansas City Chiefs':       ('AFC', 'West'),
    'Las Vegas Raiders':        ('AFC', 'West'),
    'Los Angeles Raiders':      ('AFC', 'West'),  # 1982-1994
    'Oakland Raiders':          ('AFC', 'West'),  # 1970-1981, 1995-2019
    'Los Angeles Chargers':     ('AFC', 'West'),  # since 2017
    'San Diego Chargers':       ('AFC', 'West'),

    # NFC East
    'Dallas Cowboys':           ('NFC', 'East'),
    'New York Giants':          ('NFC', 'East'),
    'Philadelphia Eagles':      ('NFC', 'East'),
    'Washington Commanders':    ('NFC', 'East'),  # since 2022
    'Washington Football Team': ('NFC', 'East'),  # 2020-2021
    'Washington Redskins':      ('NFC', 'East'),  # until 2019

    # NFC North (post-2002, was NFC Central)
    'Chicago Bears':            ('NFC', 'North'),
    'Detroit Lions':            ('NFC', 'North'),
    'Green Bay Packers':        ('NFC', 'North'),
    'Minnesota Vikings':        ('NFC', 'North'),

    # NFC South (post-2002, new division)
    'Atlanta Falcons':          ('NFC', 'South'),
    'Carolina Panthers':        ('NFC', 'South'),
    'New Orleans Saints':       ('NFC', 'South'),
    'Tampa Bay Buccaneers':     ('NFC', 'South'),

    # NFC West
    'Arizona Cardinals':        ('NFC', 'West'),  # since 1994 (was NFC East until 2002)
    'Phoenix Cardinals':        ('NFC', 'West'),  # 1988-1993; lineage to current AZ
    'St. Louis Cardinals':      ('NFC', 'West'),  # 1970-1987; lineage to current AZ
    'Los Angeles Rams':         ('NFC', 'West'),  # 1970-1994, 2016-now
    'St. Louis Rams':           ('NFC', 'West'),  # 1995-2015
    'San Francisco 49ers':      ('NFC', 'West'),
    'Seattle Seahawks':         ('NFC', 'West'),  # since 2002 (was AFC West before)
}


def conf(team):
    return TEAM_CONFERENCE.get(team, ('Other', 'Other'))[0]


def div(team):
    return TEAM_CONFERENCE.get(team, ('Other', 'Other'))[1]


# ── Era-aware display names ─────────────────────────────────────────────────
# dillon.py uses canonical (current) franchise names internally so a team's
# rating is continuous across same-market rebrands. Historical UI views
# (GOAT, Champions, Standings, per-team Season cells) should show what the
# team was actually called at the time. Maps canonical → list of
# (start_season, end_season_inclusive, display_name) ranges. 9999 = ongoing.
NFL_TEAM_DISPLAY_HISTORY = {
    'New England Patriots':  [(1970, 1970, 'Boston Patriots'),
                              (1971, 9999, 'New England Patriots')],
    'Washington Commanders': [(1970, 2019, 'Washington Redskins'),
                              (2020, 2021, 'Washington Football Team'),
                              (2022, 9999, 'Washington Commanders')],
    'Arizona Cardinals':     [(1988, 1993, 'Phoenix Cardinals'),
                              (1994, 9999, 'Arizona Cardinals')],
    'Tennessee Titans':      [(1997, 1998, 'Tennessee Oilers'),
                              (1999, 9999, 'Tennessee Titans')],
}


def display_name(canonical, season):
    """Era-appropriate display name for the given canonical team and season."""
    history = NFL_TEAM_DISPLAY_HISTORY.get(canonical)
    if not history:
        return canonical
    s = int(season)
    for start, end, name in history:
        if start <= s <= end:
            return name
    return canonical


def current_display_name(canonical):
    """The team's most recent display name (used for dropdowns / current snapshot)."""
    history = NFL_TEAM_DISPLAY_HISTORY.get(canonical)
    if not history:
        return canonical
    return history[-1][2]


def historical_display_names(canonical):
    """Prior display names (most recent first), excluding the current name.
    Used to render '(formerly X / Y)' hints in the Team Summary dropdown."""
    history = NFL_TEAM_DISPLAY_HISTORY.get(canonical)
    if not history:
        return []
    current = history[-1][2]
    seen = {current}
    out = []
    for _, _, name in reversed(history[:-1]):
        if name not in seen:
            out.append(name)
            seen.add(name)
    return out


# ── Per-(team, season) conference + division history ────────────────────────
# Era-aware lookup so historical Standings / Team Summary / Champions / GOAT
# entries show the actual conference + division a team played in that season.
#
# Two major realignments:
#   - 1976 expansion: Tampa Bay (AFC West) and Seattle (NFC West) for one year
#     only, then swapped 1977 (TB → NFC Central, SEA → AFC West).
#   - 2002 realignment: 6 divisions → 8 four-team divisions. AFC/NFC Central
#     renamed to North; AFC South + NFC South created; multiple teams moved
#     (Seahawks back to NFC, Cardinals NFC East → West, Bucs/Falcons/Saints
#     to NFC South, Oilers/Titans/Jaguars to AFC South, Ravens to AFC North).
#
# Format: team_name → list of (first_season, last_season_inclusive, conf, division).
# 9999 = ongoing. Division name has no conf prefix (e.g. 'East', not 'AFC East').
TEAM_DIVISION_HISTORY = {
    # AFC East - stable from 1970
    'Buffalo Bills':            [(1970, 9999, 'AFC', 'East')],
    'Miami Dolphins':           [(1970, 9999, 'AFC', 'East')],
    'Boston Patriots':          [(1970, 1970, 'AFC', 'East')],
    'New England Patriots':     [(1971, 9999, 'AFC', 'East')],
    'New York Jets':            [(1970, 9999, 'AFC', 'East')],
    # Colts: AFC East 1970-2001 (Baltimore until 1983, Indianapolis 1984+),
    # moved to AFC South in 2002 realignment.
    'Baltimore Colts':          [(1970, 1983, 'AFC', 'East')],
    'Indianapolis Colts':       [(1984, 2001, 'AFC', 'East'),
                                 (2002, 9999, 'AFC', 'South')],

    # AFC Central (1970-2001) → AFC North (2002+)
    'Cincinnati Bengals':       [(1970, 2001, 'AFC', 'Central'),
                                 (2002, 9999, 'AFC', 'North')],
    'Cleveland Browns':         [(1970, 1995, 'AFC', 'Central'),
                                 # 1996-1998: franchise dormant (became Ravens; reactivated 1999)
                                 (1999, 2001, 'AFC', 'Central'),
                                 (2002, 9999, 'AFC', 'North')],
    'Pittsburgh Steelers':      [(1970, 2001, 'AFC', 'Central'),
                                 (2002, 9999, 'AFC', 'North')],
    'Baltimore Ravens':         [(1996, 2001, 'AFC', 'Central'),
                                 (2002, 9999, 'AFC', 'North')],

    # AFC South (created 2002) - Houston/Tennessee lineage was AFC Central
    'Houston Oilers':           [(1970, 1996, 'AFC', 'Central')],
    'Tennessee Oilers':         [(1997, 1998, 'AFC', 'Central')],
    # 'Tennessee Titans' is the DILLON-canonical name for Tennessee Oilers
    # (1997-1998) too via the TEAM_ALIASES collapse in dillon.py, so its
    # history must cover the AFC-Central Oilers era as well - otherwise
    # 1997/1998 'Tennessee Titans' rows fall through to the modern AFC South
    # fallback and corrupt playoff-field derivation.
    'Tennessee Titans':         [(1997, 2001, 'AFC', 'Central'),
                                 (2002, 9999, 'AFC', 'South')],
    'Jacksonville Jaguars':     [(1995, 2001, 'AFC', 'Central'),
                                 (2002, 9999, 'AFC', 'South')],
    'Houston Texans':           [(2002, 9999, 'AFC', 'South')],

    # AFC West - stable, with relocations within division
    'Denver Broncos':           [(1970, 9999, 'AFC', 'West')],
    'Kansas City Chiefs':       [(1970, 9999, 'AFC', 'West')],
    'Oakland Raiders':          [(1970, 1981, 'AFC', 'West'),
                                 (1995, 2019, 'AFC', 'West')],
    'Los Angeles Raiders':      [(1982, 1994, 'AFC', 'West')],
    'Las Vegas Raiders':        [(2020, 9999, 'AFC', 'West')],
    'San Diego Chargers':       [(1970, 2016, 'AFC', 'West')],
    'Los Angeles Chargers':     [(2017, 9999, 'AFC', 'West')],
    # Seahawks: 1976 NFC West expansion year, 1977-2001 AFC West, 2002+ NFC West
    'Seattle Seahawks':         [(1976, 1976, 'NFC', 'West'),
                                 (1977, 2001, 'AFC', 'West'),
                                 (2002, 9999, 'NFC', 'West')],

    # NFC East - stable from 1970
    'Dallas Cowboys':           [(1970, 9999, 'NFC', 'East')],
    'New York Giants':          [(1970, 9999, 'NFC', 'East')],
    'Philadelphia Eagles':      [(1970, 9999, 'NFC', 'East')],
    'Washington Redskins':      [(1970, 2019, 'NFC', 'East')],
    'Washington Football Team': [(2020, 2021, 'NFC', 'East')],
    'Washington Commanders':    [(2022, 9999, 'NFC', 'East')],
    # Cardinals: NFC East 1970-2001 (St. Louis until 1987, Phoenix 1988-1993,
    # Arizona 1994+), moved to NFC West in 2002 realignment.
    'St. Louis Cardinals':      [(1970, 1987, 'NFC', 'East')],
    'Phoenix Cardinals':        [(1988, 1993, 'NFC', 'East')],
    # DILLON's TEAM_ALIASES collapses Phoenix Cardinals into 'Arizona
    # Cardinals', so its history must cover the Phoenix-era NFC East years
    # too (otherwise 1988-1993 rows fall through to the modern NFC West
    # fallback).
    'Arizona Cardinals':        [(1988, 2001, 'NFC', 'East'),
                                 (2002, 9999, 'NFC', 'West')],

    # NFC Central (1970-2001) → NFC North (2002+)
    'Chicago Bears':            [(1970, 2001, 'NFC', 'Central'),
                                 (2002, 9999, 'NFC', 'North')],
    'Detroit Lions':            [(1970, 2001, 'NFC', 'Central'),
                                 (2002, 9999, 'NFC', 'North')],
    'Green Bay Packers':        [(1970, 2001, 'NFC', 'Central'),
                                 (2002, 9999, 'NFC', 'North')],
    'Minnesota Vikings':        [(1970, 2001, 'NFC', 'Central'),
                                 (2002, 9999, 'NFC', 'North')],
    # Bucs: 1976 AFC West expansion year, 1977-2001 NFC Central, 2002+ NFC South
    'Tampa Bay Buccaneers':     [(1976, 1976, 'AFC', 'West'),
                                 (1977, 2001, 'NFC', 'Central'),
                                 (2002, 9999, 'NFC', 'South')],

    # NFC South (created 2002) - Falcons/Saints were NFC West, Bucs NFC Central
    'Atlanta Falcons':          [(1970, 2001, 'NFC', 'West'),
                                 (2002, 9999, 'NFC', 'South')],
    'Carolina Panthers':        [(1995, 2001, 'NFC', 'West'),
                                 (2002, 9999, 'NFC', 'South')],
    'New Orleans Saints':       [(1970, 2001, 'NFC', 'West'),
                                 (2002, 9999, 'NFC', 'South')],

    # NFC West - Rams + 49ers (and Seahawks 1976, Bucs/etc handled above)
    'Los Angeles Rams':         [(1970, 1994, 'NFC', 'West'),
                                 (2016, 9999, 'NFC', 'West')],
    'St. Louis Rams':           [(1995, 2015, 'NFC', 'West')],
    'San Francisco 49ers':      [(1970, 9999, 'NFC', 'West')],
}


def _conf_div_for(team, season):
    s = int(season)
    history = TEAM_DIVISION_HISTORY.get(team)
    if history:
        for start, end, c, d in history:
            if start <= s <= end:
                return (c, d)
    # Fallback: current-era mapping (for any team not yet in history dict).
    return TEAM_CONFERENCE.get(team, ('Other', 'Other'))


def conf_for_season(team, season):
    return _conf_div_for(team, season)[0]


def div_for_season(team, season):
    return _conf_div_for(team, season)[1]


def clean(val):
    if pd.isna(val):
        return ''
    return str(val)


# dillon.py constructs last_match strings using the canonical franchise name
# (e.g. "W 27-20 vs. Washington Commanders" for a 1995 game when the team
# was actually the Redskins). Rewrite the opponent portion with the era-
# appropriate display name so historical Team Summary / Standings views show
# the franchise's contemporary name. Handles `vs.`, `@`, and `vs. (N)`
# (neutral-site) separators.
_LAST_MATCH_RE = re.compile(r'^([WLT])\s+(\d+\s*-\s*\d+)\s+(vs\.?(?:\s*\(N\))?|@)\s+(.+)$')

def era_aware_last_match(raw, season):
    if not raw:
        return raw
    m = _LAST_MATCH_RE.match(str(raw))
    if not m:
        return raw
    letter, score, venue, opponent = m.groups()
    return f"{letter} {score} {venue} {display_name(opponent.strip(), season)}"


def slug(name):
    return re.sub(r'[^\w]', '_', name).strip('_')


def _played(result):
    """True iff this row represents an actual game played. Upstream now
    writes empty strings for non-game-days (was 'Bye / No Game' previously)
    - both must be treated as "didn't play" or the forward-fill of
    last_match breaks for any week a team had a bye."""
    if result is None or pd.isna(result):
        return False
    s = str(result).strip()
    return s not in ('', 'Bye / No Game')


# is_game_day: row where the team actually played that week
df['is_game_day'] = df['lastgame'].apply(_played).astype(int)
df['is_end_of_season'] = df['season_flag'].isin([1, 2]).astype(int)

# Per-(team, season) forward-filled last game. Keying by season prevents
# cross-season carry-forward - at the start of a new season, teams that
# haven't played yet correctly show empty rather than their previous-season
# Super Bowl result.
_last_game_history = {}
for (team, season), tdf in df[df['is_game_day'] == 1].sort_values('season_week').groupby(['name', 'season']):
    _last_game_history[(team, int(season))] = (
        list(tdf['season_week']),
        list(tdf['lastgame']),
        list(tdf['date']),
    )


def last_game_as_of(team, sw, season):
    entry = _last_game_history.get((team, int(season)))
    if not entry:
        return ''
    sws, games_list, _ = entry
    idx = bisect_right(sws, sw) - 1
    return games_list[idx] if idx >= 0 else ''


def last_game_date_as_of(team, sw, season):
    entry = _last_game_history.get((team, int(season)))
    if not entry:
        return ''
    sws, _, dates = entry
    idx = bisect_right(sws, sw) - 1
    return dates[idx] if idx >= 0 else ''


# Per-season last regular-season week
_rs_end_sw = (
    df[df['season_flag'] == 1]
    .groupby('season')['season_week']
    .max()
    .to_dict()
)


def is_playoff(season, sw):
    rs_end = _rs_end_sw.get(season)
    if rs_end is None:
        # Fall back to week-based detection (week >= 100 = playoff)
        return False
    return sw > rs_end


# Regular-season-end record per (team, season) from season_flag == 1 snapshots
_reg_record_lookup = {
    (row['name'], int(row['season'])): row['record']
    for _, row in df[df['season_flag'] == 1].iterrows()
}

# End-of-regular-season rating per (team, season) from season_flag == 1
# snapshots. The champion entry's own `rating` is the end-of-playoffs (flag == 2)
# rating, so pairing the two gives the Lists sub-view its "biggest playoff leap"
# ranking (playoffs - regular season).
_rs_rating_lookup = {
    (row['name'], int(row['season'])): round(float(row['rating']), 3)
    for _, row in df[df['season_flag'] == 1].iterrows()
}

# Final (post-playoff) record per (team, season) from season_flag == 2 snapshots.
# Used so the GOAT-RS view can show each team's actual playoff outcome alongside
# their regular-season record - mirrors GRIFFEY/SAKIC/COBI, where playoff_record
# in goat_rs reflects how the team's playoffs ultimately went. Teams that
# didn't make the playoffs fall back to their regular_record (giving "0-0").
_final_record_lookup = {
    (row['name'], int(row['season'])): row['record']
    for _, row in df[df['season_flag'] == 2].iterrows()
}


def _parse_record(rec):
    """Parse 'W-L' or 'W-L-T'. Returns (wins, losses, ties)."""
    if not rec or pd.isna(rec):
        return None
    parts = str(rec).split('-')
    try:
        if len(parts) == 2:
            return int(parts[0]), int(parts[1]), 0
        if len(parts) == 3:
            return int(parts[0]), int(parts[1]), int(parts[2])
    except ValueError:
        return None
    return None


def playoff_record(full_record, regular_record):
    """Compute playoff record as (full - regular). Playoffs cannot have ties."""
    f = _parse_record(full_record)
    r = _parse_record(regular_record)
    if not f or not r:
        return ''
    pw = f[0] - r[0]
    pl = f[1] - r[1]
    if pw < 0 or pl < 0:
        return ''
    return f"{pw}-{pl}"


WEEK_LABELS = {
    101: 'Wild Card',
    102: 'Divisional',
    103: 'Conf Champ',
    104: 'Super Bowl',
}


def week_label(wk):
    return WEEK_LABELS.get(int(wk), f'Week {int(wk)}')


def snapshot_label(wk, flag):
    base = week_label(wk)
    if flag == 1:
        return f'{base} · End of regular season'
    if flag == 2:
        return f'{base} · End of playoffs'
    return base


def _record_pct(rec):
    """Win pct from W-L or W-L-T record string. Ties count as half-wins."""
    p = _parse_record(rec)
    if not p:
        return -1.0
    w, l, t = p
    g = w + l + t
    return (w + 0.5 * t) / g if g > 0 else 0.0


# Regular-season-only games (decimal week < 0.101) for tiebreaker lookups.
_rs_games = games[games['season_week'] - games['season'] < 0.101].copy()


def _h2h_pct(season, team, opponents, rs_games):
    """Win pct in head-to-head games between `team` and any team in `opponents`."""
    sub = rs_games[
        (rs_games['season'] == season)
        & ((rs_games['home_team_name'] == team) | (rs_games['visitor_team_name'] == team))
        & ((rs_games['home_team_name'].isin(opponents)) | (rs_games['visitor_team_name'].isin(opponents)))
    ]
    # Exclude self-vs-self (defensive; shouldn't happen).
    sub = sub[~((sub['home_team_name'] == team) & (sub['visitor_team_name'] == team))]
    if sub.empty:
        return None
    w = ((sub['winner'] == team) & (sub['is_tie'] == 0)).sum()
    ties = (sub['is_tie'] == 1).sum() if 'is_tie' in sub.columns else 0
    # Treat ties as half a win per NFL convention.
    games_played = len(sub)
    return (w + 0.5 * ties) / games_played if games_played else None


def _common_opponents(season, teams, rs_games):
    """Intersection of every team's opponent set this season. None if <4 common."""
    s = int(season)
    season_games = rs_games[rs_games['season'] == s]
    if season_games.empty:
        return None
    team_opps = {}
    for t in teams:
        opps = set()
        opps.update(season_games[season_games['home_team_name'] == t]['visitor_team_name'])
        opps.update(season_games[season_games['visitor_team_name'] == t]['home_team_name'])
        opps.discard(t)
        # Also exclude the tied teams themselves - common-games is vs OTHER common opps.
        opps -= set(teams)
        team_opps[t] = opps
    if not team_opps:
        return None
    common = set.intersection(*team_opps.values())
    if len(common) < 4:
        return None
    return common


def _common_games_pct(season, team, common_opps, rs_games):
    if not common_opps:
        return None
    return _h2h_pct(season, team, common_opps, rs_games)


# Pre-compute every team's full-season W-L% across all their RS games,
# used by Strength-of-Victory and Strength-of-Schedule tiebreakers.
def _build_season_records(season, rs_games):
    s = int(season)
    season_games = rs_games[rs_games['season'] == s]
    teams = set(season_games['home_team_name']) | set(season_games['visitor_team_name'])
    records = {}
    for t in teams:
        team_games = season_games[
            (season_games['home_team_name'] == t) | (season_games['visitor_team_name'] == t)
        ]
        if team_games.empty:
            continue
        w = ((team_games['winner'] == t) & (team_games['is_tie'] == 0)).sum()
        ties = (team_games['is_tie'] == 1).sum() if 'is_tie' in team_games.columns else 0
        n = len(team_games)
        records[t] = (w + 0.5 * ties) / n if n > 0 else 0.0
    return records


def _strength_of_victory(season, team, season_records, rs_games):
    """Avg W-L% of teams `team` beat this season."""
    s = int(season)
    season_games = rs_games[rs_games['season'] == s]
    won = season_games[
        ((season_games['home_team_name'] == team) & (season_games['home_pts'] > season_games['visitor_pts']))
        | ((season_games['visitor_team_name'] == team) & (season_games['visitor_pts'] > season_games['home_pts']))
    ]
    opps = []
    for _, g in won.iterrows():
        opp = g['visitor_team_name'] if g['home_team_name'] == team else g['home_team_name']
        opps.append(opp)
    if not opps:
        return 0.0
    pcts = [season_records.get(o, 0.0) for o in opps]
    return sum(pcts) / len(pcts)


def _strength_of_schedule(season, team, season_records, rs_games):
    """Avg W-L% of every opponent `team` played this season (won or lost)."""
    s = int(season)
    season_games = rs_games[rs_games['season'] == s]
    played = season_games[
        (season_games['home_team_name'] == team) | (season_games['visitor_team_name'] == team)
    ]
    opps = []
    for _, g in played.iterrows():
        opp = g['visitor_team_name'] if g['home_team_name'] == team else g['home_team_name']
        opps.append(opp)
    if not opps:
        return 0.0
    pcts = [season_records.get(o, 0.0) for o in opps]
    return sum(pcts) / len(pcts)


_pts_rankings_cache = {}


def _build_pts_rankings(season, rs_games):
    """Build per-team points-for/points-against totals and the
    combined-ranking values used by NFL tiebreaker steps 7-8."""
    s = int(season)
    season_games = rs_games[rs_games['season'] == s]
    all_teams = set(season_games['home_team_name']) | set(season_games['visitor_team_name'])

    pts_all = {t: [0, 0] for t in all_teams}  # [PS, PA]
    pts_conf = {t: [0, 0] for t in all_teams}

    for _, g in season_games.iterrows():
        h, v = g['home_team_name'], g['visitor_team_name']
        h_pts, v_pts = int(g['home_pts']), int(g['visitor_pts'])
        same_conf = conf_for_season(h, s) == conf_for_season(v, s)
        if h in pts_all:
            pts_all[h][0] += h_pts
            pts_all[h][1] += v_pts
            if same_conf:
                pts_conf[h][0] += h_pts
                pts_conf[h][1] += v_pts
        if v in pts_all:
            pts_all[v][0] += v_pts
            pts_all[v][1] += h_pts
            if same_conf:
                pts_conf[v][0] += v_pts
                pts_conf[v][1] += h_pts

    by_conf = collections.defaultdict(list)
    for t in all_teams:
        by_conf[conf_for_season(t, s)].append(t)

    rankings = {}
    # Conference combined PS+PA rank (lower is better)
    for conf, teams in by_conf.items():
        ps_sorted = sorted(teams, key=lambda t: -pts_conf[t][0])
        ps_rank = {t: i + 1 for i, t in enumerate(ps_sorted)}
        pa_sorted = sorted(teams, key=lambda t: pts_conf[t][1])
        pa_rank = {t: i + 1 for i, t in enumerate(pa_sorted)}
        for t in teams:
            rankings.setdefault(t, {})['conf_combined'] = ps_rank[t] + pa_rank[t]

    # League-wide combined PS+PA rank
    all_list = list(all_teams)
    ps_sorted = sorted(all_list, key=lambda t: -pts_all[t][0])
    ps_rank = {t: i + 1 for i, t in enumerate(ps_sorted)}
    pa_sorted = sorted(all_list, key=lambda t: pts_all[t][1])
    pa_rank = {t: i + 1 for i, t in enumerate(pa_sorted)}
    for t in all_list:
        rankings.setdefault(t, {})['all_combined'] = ps_rank[t] + pa_rank[t]
        rankings[t]['pts_all'] = tuple(pts_all[t])
        rankings[t]['pts_conf'] = tuple(pts_conf[t])

    return rankings


def _get_pts_rankings(season, rs_games):
    if season not in _pts_rankings_cache:
        _pts_rankings_cache[season] = _build_pts_rankings(season, rs_games)
    return _pts_rankings_cache[season]


def _common_games_net_pts(season, team, common_opps, rs_games):
    """Net points (scored - allowed) in games vs common opponents."""
    if not common_opps:
        return 0
    s = int(season)
    season_games = rs_games[rs_games['season'] == s]
    net = 0
    for _, g in season_games.iterrows():
        if g['home_team_name'] == team and g['visitor_team_name'] in common_opps:
            net += int(g['home_pts']) - int(g['visitor_pts'])
        elif g['visitor_team_name'] == team and g['home_team_name'] in common_opps:
            net += int(g['visitor_pts']) - int(g['home_pts'])
    return net


def _apply_points_based_cascade(season, tied_teams, rs_games):
    """Steps 7-10 of NFL tiebreaker cascade (shared by division + wild card):
      7. Conference combined PS+PA ranking (lower is better)
      8. League combined PS+PA ranking
      9. Best net points in common games (>=4 common opponents)
     10. Best net points in all games
     Then alphabetical fallback. Returns single winner."""
    survivors = list(tied_teams)
    if len(survivors) == 1:
        return survivors[0]
    rankings = _get_pts_rankings(int(season), rs_games)

    # 7. Conference combined ranking (lower = better)
    cc = {t: rankings.get(t, {}).get('conf_combined', 999) for t in survivors}
    best = min(cc.values())
    survivors2 = [t for t in survivors if cc[t] == best]
    if 0 < len(survivors2) < len(survivors):
        if len(survivors2) == 1: return survivors2[0]
        survivors = survivors2

    # 8. All-games combined ranking
    ac = {t: rankings.get(t, {}).get('all_combined', 999) for t in survivors}
    best = min(ac.values())
    survivors2 = [t for t in survivors if ac[t] == best]
    if 0 < len(survivors2) < len(survivors):
        if len(survivors2) == 1: return survivors2[0]
        survivors = survivors2

    # 9. Net points common games
    common = _common_opponents(int(season), survivors, rs_games)
    if common:
        np_common = {t: _common_games_net_pts(int(season), t, common, rs_games) for t in survivors}
        best = max(np_common.values())
        survivors2 = [t for t in survivors if np_common[t] == best]
        if 0 < len(survivors2) < len(survivors):
            if len(survivors2) == 1: return survivors2[0]
            survivors = survivors2

    # 10. Net points all games
    np_all = {t: rankings.get(t, {}).get('pts_all', (0, 0))[0]
              - rankings.get(t, {}).get('pts_all', (0, 0))[1] for t in survivors}
    best = max(np_all.values())
    survivors2 = [t for t in survivors if np_all[t] == best]
    if 0 < len(survivors2) < len(survivors):
        if len(survivors2) == 1: return survivors2[0]
        survivors = survivors2

    return sorted(survivors)[0]


def _div_record_pct(season, team, rs_games):
    """Win pct in the team's intra-division games this season."""
    div_peers = {
        t for t in rs_games[rs_games['season'] == season]['home_team_name'].unique()
        if conf_for_season(t, season) == conf_for_season(team, season)
        and div_for_season(t, season) == div_for_season(team, season)
        and t != team
    }
    if not div_peers:
        return None
    return _h2h_pct(season, team, div_peers, rs_games)


def _conf_record_pct(season, team, rs_games):
    """Win pct in the team's intra-conference games this season."""
    conf_peers = {
        t for t in rs_games[rs_games['season'] == season]['home_team_name'].unique()
        if conf_for_season(t, season) == conf_for_season(team, season)
        and t != team
    }
    if not conf_peers:
        return None
    return _h2h_pct(season, team, conf_peers, rs_games)


def _resolve_division_tie(season, tied_teams, rs_games):
    """Break a division tie using the NFL division-tiebreaker cascade:
       1. Head-to-head
       2. Best record in division games
       3. Best record in common games (>=4 common opponents)
       4. Best record in conference games
       5. Strength of victory
       6. Strength of schedule
       7. Alphabetical fallback
    Returns the winning team name."""
    if len(tied_teams) == 1:
        return tied_teams[0]
    s_int = int(season)

    def _apply(pcts):
        if not pcts or any(v is None for v in pcts.values()):
            return None
        best = max(pcts.values())
        survivors = [t for t in tied_teams if pcts[t] == best]
        if 0 < len(survivors) < len(tied_teams):
            return survivors
        return None

    # 1) Head-to-head
    h2h = {t: _h2h_pct(s_int, t, [x for x in tied_teams if x != t], rs_games) for t in tied_teams}
    survivors = _apply(h2h)
    if survivors is not None:
        if len(survivors) == 1:
            return survivors[0]
        tied_teams = survivors

    # 2) Division record
    drec = {t: _div_record_pct(s_int, t, rs_games) for t in tied_teams}
    survivors = _apply(drec)
    if survivors is not None:
        if len(survivors) == 1:
            return survivors[0]
        tied_teams = survivors

    # 3) Common games (>=4 common opponents)
    common = _common_opponents(s_int, tied_teams, rs_games)
    if common:
        cg = {t: _common_games_pct(s_int, t, common, rs_games) for t in tied_teams}
        survivors = _apply(cg)
        if survivors is not None:
            if len(survivors) == 1:
                return survivors[0]
            tied_teams = survivors

    # 4) Conference record
    crec = {t: _conf_record_pct(s_int, t, rs_games) for t in tied_teams}
    survivors = _apply(crec)
    if survivors is not None:
        if len(survivors) == 1:
            return survivors[0]
        tied_teams = survivors

    # 5) Strength of victory
    season_records = _season_records_cache.get(s_int)
    if season_records is None:
        season_records = _build_season_records(s_int, rs_games)
        _season_records_cache[s_int] = season_records
    sov = {t: _strength_of_victory(s_int, t, season_records, rs_games) for t in tied_teams}
    survivors = _apply(sov)
    if survivors is not None:
        if len(survivors) == 1:
            return survivors[0]
        tied_teams = survivors

    # 6) Strength of schedule
    sos = {t: _strength_of_schedule(s_int, t, season_records, rs_games) for t in tied_teams}
    survivors = _apply(sos)
    if survivors is not None:
        if len(survivors) == 1:
            return survivors[0]
        tied_teams = survivors

    # 7-10) Points-based cascade (combined ranking, net points)
    return _apply_points_based_cascade(s_int, tied_teams, rs_games)


_season_records_cache = {}


# ── Division winners (per season + (conference, division)) ───────────────────
# Tag the team with the best RS record in each (conference, division) at end
# of regular season. Real NFL tiebreakers are 9 steps deep - we apply the first
# three (head-to-head → division record → conference record) which resolve the
# vast majority of historical ties; alphabetical name is the final fallback.
_division_winners = set()  # set of (season, team) tuples
for season, sub in df[df['season_flag'] == 1].groupby('season'):
    s_int = int(season)
    sub = sub.copy()
    sub['_conf'] = sub.apply(lambda r: conf_for_season(r['name'], s_int), axis=1)
    sub['_div']  = sub.apply(lambda r: div_for_season(r['name'], s_int), axis=1)
    sub = sub[(sub['_conf'] != 'Other') & (sub['_div'] != 'Other')]
    sub['_pct']  = sub['record'].apply(_record_pct)
    for (_cf, _dv), grp in sub.groupby(['_conf', '_div']):
        top = grp['_pct'].max()
        tied = grp[grp['_pct'] == top]['name'].tolist()
        if not tied:
            continue
        winner = tied[0] if len(tied) == 1 else _resolve_division_tie(s_int, tied, _rs_games)
        _division_winners.add((s_int, winner))
print(f"  {len(_division_winners)} division winners flagged.")


# ── Team-specific home field advantage (3-year rolling-day window) ───────────
# For each team T and each snapshot R (dated at snap_date), compute T's home
# advantage as the GAP between T's home-game and away-game performance, where
# "performance" is residual against a neutral-site prediction (rating gap with
# no HCA term):
#
#   home_resid(G, T)     = (home_pts - away_pts) - (T_rating - opp_rating)        when T is home
#   away_resid_T_POV(G,T) = -(home_pts - away_pts) - (T_rating - opp_rating)        when T is away
#
#   HFA_T(R) = (mean(home_resid over T's home games in window)
#               - mean(away_resid_T_POV over T's away games in window)) / 2
#
# Window: games with date in (snap_date - 1095 days, snap_date]. True rolling-
# day window - every week a new game enters the front AND a game from exactly
# 3 years ago drops off the back. So HFA updates every snapshot regardless of
# whether T played at home or away that week.
#
# Every game contributes signal: a team's home games tell us how much they
# outperform expectation when hosting; their away games tell us how much they
# underperform when traveling. The gap is the home advantage. League average
# lands near the global HCA constant (~2.5); altitude / cold-weather / loud-
# dome teams sit above, soft-market teams below.

print("Computing per-snapshot home field advantage (3-year rolling-day window)...")
_HFA_DAYS = 3 * 365  # 1095-day rolling window
_MIN_HOME_GAMES = 10  # below this, hfa estimate is too noisy; suppress

_rsorted = df.sort_values(['name', 'ranking_id']).copy()
_rsorted['rating_prior'] = _rsorted.groupby('name')['rating'].shift(1)
_rating_prior_lookup = _rsorted.set_index(['name', 'season', 'week'])['rating_prior'].to_dict()

def _prior_rating(name, season, week):
    return _rating_prior_lookup.get((name, int(season), int(week)))

# Build the full hfa_contribution table - one row per non-neutral game with
# valid prior ratings for both teams. Reused across all snapshot lookups.
_all_hfa_games = games[games['is_neutral'] == 0].copy()
_all_hfa_games['home_rating'] = _all_hfa_games.apply(
    lambda g: _prior_rating(g['home_team_name'], g['season'], g['week']), axis=1)
_all_hfa_games['away_rating'] = _all_hfa_games.apply(
    lambda g: _prior_rating(g['visitor_team_name'], g['season'], g['week']), axis=1)
_all_hfa_games = _all_hfa_games.dropna(subset=['home_rating', 'away_rating']).copy()
# Per-game home-team residual at neutral expectation. Positive = home outperformed.
# Away team's residual from their POV is the negation of this value.
_all_hfa_games['home_resid'] = (
    (_all_hfa_games['home_pts'] - _all_hfa_games['visitor_pts'])
    - (_all_hfa_games['home_rating'] - _all_hfa_games['away_rating'])
)
# Pre-cast date so window filters are fast.
_all_hfa_games['date'] = pd.to_datetime(_all_hfa_games['date'])
print(f"  HFA-eligible games: {len(_all_hfa_games):,}")

# Snapshot date lookup so _hfa_snapshot can find its window start.
_snapshot_dates = (
    df.dropna(subset=['date'])
    .groupby('ranking_id')['date']
    .first()
    .to_dict()
)


def _hfa_snapshot(ranking_id):
    """Return {team: {'hfa': float, 'rank': int}} for the 1095-day rolling
    window ending at this snapshot's date. Uses both home games (positive
    residual contribution) and away games (residual from the traveling team's
    POV = -home_resid). HFA = (home_mean - away_mean) / 2."""
    snap_date_str = _snapshot_dates.get(ranking_id)
    if not snap_date_str:
        return {}
    snap_date = pd.to_datetime(snap_date_str)
    window_start = snap_date - pd.Timedelta(days=_HFA_DAYS)
    win = _all_hfa_games[
        (_all_hfa_games['date'] > window_start)
        & (_all_hfa_games['date'] <= snap_date)
    ]
    if win.empty:
        return {}

    # Home-team residuals (T's POV when at home).
    home_stats = win.groupby('home_team_name')['home_resid'].agg(['mean', 'size'])
    home_stats.columns = ['home_mean', 'home_n']

    # Away-team residuals from T's POV (negated home_resid).
    away_view = win.assign(t_pov_resid=-win['home_resid'])
    away_stats = away_view.groupby('visitor_team_name')['t_pov_resid'].agg(['mean', 'size'])
    away_stats.columns = ['away_mean', 'away_n']

    combined = home_stats.join(away_stats, how='outer')
    combined = combined.fillna({'home_mean': 0.0, 'home_n': 0, 'away_mean': 0.0, 'away_n': 0})
    # Require enough home games for the home-mean to be stable.
    combined = combined[combined['home_n'] >= _MIN_HOME_GAMES].copy()
    if combined.empty:
        return {}

    combined['hfa'] = ((combined['home_mean'] - combined['away_mean']) / 2.0).round(2)
    combined['rank'] = combined['hfa'].rank(ascending=False, method='min').astype(int)
    return {team: {'hfa': float(row['hfa']), 'rank': int(row['rank'])}
            for team, row in combined.iterrows()}

# Pre-compute snapshot HFA for every ranking_id so per-season + per-team
# writers share the cache. ~1100 snapshot computations; runs in seconds.
_LATEST_SEASON = int(df['season'].max())
_LATEST_RID = int(df['ranking_id'].max())

_snapshot_hfa_cache = {}  # ranking_id -> {team: {hfa, rank}}
for _rid in sorted(_snapshot_dates):
    _snapshot_hfa_cache[int(_rid)] = _hfa_snapshot(int(_rid))

# Latest-snapshot HFA used by current_standings.json + teams_index.json.
_hfa_lookup = _snapshot_hfa_cache.get(_LATEST_RID, {})

def _hfa_val(snap_hfa, team):
    """Extract just the hfa value from a snapshot's hfa dict; None if missing."""
    rec = snap_hfa.get(team)
    return rec['hfa'] if rec else None

def _hfa_rk(snap_hfa, team):
    """Extract the hfa rank from a snapshot's hfa dict; None if missing."""
    rec = snap_hfa.get(team)
    return rec['rank'] if rec else None

_hfa_window_label = f"last {_HFA_DAYS} days"
print(f"  Cached HFA for {len(_snapshot_hfa_cache):,} snapshots; "
      f"latest snapshot has HFA for {len(_hfa_lookup)} teams "
      f"(window: {_hfa_window_label})")


# ── Super Bowl odds (logistic regression, per-week leave-one-season-out) ────
# For each team T and each regular-season snapshot R, compute P(T wins Super
# Bowl that season) by training a logistic regression on every other season's
# week-W snapshots (features: rating_o, rating_d, hfa; label: won_sb).
# Predictions for the held-out season are then normalized so the league total
# = 100% (since exactly one team wins each year).
#
# Leave-one-season-out keeps historical predictions honest: when we compute
# 2007 Giants' week-12 SB odds, the model has not been told the 2007 outcome.
# This is what makes retrospective "biggest upset" stories meaningful.
#
# Playoff snapshots (weeks 101-104) are excluded - the path is being
# revealed game-by-game and rating-based prediction stops being meaningful.

print("Computing Super Bowl odds (per-week logistic regression)...")
from scipy.optimize import minimize

REGULAR_SEASON_WEEKS = list(range(1, 19))  # weeks 1-18 (pre-2021 used 1-17)


# Mathematical playoff elimination check - used to zero out teams whose
# record makes a playoff berth literally impossible. A team is eliminated
# iff BOTH conditions hold:
#   - conf-teams-currently-ahead-of-T's-max-wins >= playoff_seeds  (can't be top-7)
#   - 1+ division opponent currently has wins > T's max wins         (can't win division)
# Winning a division earns a top-4 seed regardless of overall record, so
# both paths must be blocked for elimination. This is conservative - it
# only flags definite eliminations, never false positives.

def _total_regular_season_games(season):
    if season == 1982: return 9    # strike-shortened
    if season == 1987: return 15   # strike + cancelled week
    if season <= 1977: return 14
    if season >= 2021: return 17
    return 16

def _playoff_seeds_per_conf(season):
    if season == 1982: return 8    # expanded SB Tournament
    if season <= 1977: return 4
    if season <= 1989: return 5
    if season <= 2019: return 6
    return 7

def _parse_wlt(rec):
    if not isinstance(rec, str) or not rec:
        return 0, 0, 0
    parts = rec.split('-')
    try:
        if len(parts) == 2:
            return int(parts[0]), int(parts[1]), 0
        if len(parts) == 3:
            return int(parts[0]), int(parts[1]), int(parts[2])
    except ValueError:
        pass
    return 0, 0, 0


def _eliminated_teams_at_snapshot(season, snap_df):
    total_games = _total_regular_season_games(season)
    seeds = _playoff_seeds_per_conf(season)

    info = []
    for _, r in snap_df.iterrows():
        w, l, t = _parse_wlt(r.get('record', ''))
        gp = w + l + t
        gr = max(0, total_games - gp)
        eff_w = w + 0.5 * t
        info.append({
            'team': r['name'],
            'conf': conf_for_season(r['name'], season),
            'div':  div_for_season(r['name'], season),
            'wins': eff_w,
            'max_wins': eff_w + gr,
        })

    eliminated = set()
    for t_info in info:
        T_max = t_info['max_wins']
        ahead_conf = sum(1 for x in info
                          if x['conf'] == t_info['conf']
                          and x['team'] != t_info['team']
                          and x['wins'] > T_max)
        ahead_div = sum(1 for x in info
                         if x['conf'] == t_info['conf']
                         and x['div']  == t_info['div']
                         and x['team'] != t_info['team']
                         and x['wins'] > T_max)
        if ahead_conf >= seeds and ahead_div >= 1:
            eliminated.add(t_info['team'])
    return eliminated


# Pre-compute eliminated set per ranking_id (RS snapshots only). Cheap.
print("  Computing mathematical playoff elimination per snapshot...")
_eliminated_cache = {}
for _rid in df[df['week'].isin(REGULAR_SEASON_WEEKS)]['ranking_id'].unique():
    _snap = df[df['ranking_id'] == _rid]
    _season_val = int(_snap['season'].iloc[0])
    _elim = _eliminated_teams_at_snapshot(_season_val, _snap)
    if _elim:
        _eliminated_cache[int(_rid)] = _elim
print(f"  Eliminated-team flags cached for {len(_eliminated_cache):,} snapshots")

# Identify SB winners by season - the team flagged sb_champ at week 104.
_sb_winners = {}
for _season, _sdf in df.groupby('season'):
    _wk104 = _sdf[(_sdf['week'] == 104) & (_sdf['sb_champ'] == 1)]
    if not _wk104.empty:
        _sb_winners[int(_season)] = _wk104['name'].iloc[0]
print(f"  SB winners identified: {len(_sb_winners)} seasons")

# Build training rows: one per (team, season, regular-season-week) with valid
# (O, D, HFA) features and a known SB outcome. Skip in-progress seasons.
_sb_rows = []
for _, _r in df[df['week'].isin(REGULAR_SEASON_WEEKS)].iterrows():
    _season = int(_r['season'])
    if _season not in _sb_winners:
        continue
    if pd.isna(_r.get('rating_o')) or pd.isna(_r.get('rating_d')):
        continue
    _hfa = _hfa_val(_snapshot_hfa_cache.get(int(_r['ranking_id']), {}), _r['name'])
    if _hfa is None:
        continue
    _sb_rows.append({
        'season':     _season,
        'week':       int(_r['week']),
        'ranking_id': int(_r['ranking_id']),
        'team':       _r['name'],
        'rating_o':   float(_r['rating_o']),
        'rating_d':   float(_r['rating_d']),
        'hfa':        float(_hfa),
        'won_sb':     1 if _r['name'] == _sb_winners[_season] else 0,
    })
_sb_train_df = pd.DataFrame(_sb_rows)
print(f"  Training rows: {len(_sb_train_df):,} (team, season, regular-season-week)")


def _fit_logistic(X, y):
    """Plain logistic regression via scipy BFGS. Returns beta vector with
    intercept as first element. X shape (n, k), y shape (n,)."""
    n, k = X.shape
    Xa = np.column_stack([np.ones(n), X])

    def nll(beta):
        z = Xa @ beta
        # Numerically stable log(1 + exp(z))
        return float(np.sum(np.maximum(z, 0.0) + np.log1p(np.exp(-np.abs(z))) - y * z))

    def grad(beta):
        z = Xa @ beta
        p_hat = 1.0 / (1.0 + np.exp(-z))
        return Xa.T @ (p_hat - y)

    res = minimize(nll, np.zeros(k + 1), jac=grad, method='BFGS',
                   options={'maxiter': 200, 'gtol': 1e-6})
    return res.x


def _predict_logistic(X, beta):
    n = X.shape[0]
    Xa = np.column_stack([np.ones(n), X])
    z = Xa @ beta
    return 1.0 / (1.0 + np.exp(-z))


# Build the also-in-progress-current-season frame: for seasons NOT in
# _sb_winners (in-progress), we'll still predict using a model trained on
# all known seasons.
_current_rows = []
for _, _r in df[df['week'].isin(REGULAR_SEASON_WEEKS)].iterrows():
    _season = int(_r['season'])
    if _season in _sb_winners:
        continue  # handled in LOO loop
    if pd.isna(_r.get('rating_o')) or pd.isna(_r.get('rating_d')):
        continue
    _hfa = _hfa_val(_snapshot_hfa_cache.get(int(_r['ranking_id']), {}), _r['name'])
    if _hfa is None:
        continue
    _current_rows.append({
        'season':     _season,
        'week':       int(_r['week']),
        'ranking_id': int(_r['ranking_id']),
        'team':       _r['name'],
        'rating_o':   float(_r['rating_o']),
        'rating_d':   float(_r['rating_d']),
        'hfa':        float(_hfa),
    })
_sb_current_df = pd.DataFrame(_current_rows)

# Fit + predict
_sb_odds_cache = {}  # (ranking_id, team) -> sb_odds  (float, 0-1)
_FEATURES = ['rating_o', 'rating_d', 'hfa']

for _week_val, _week_data in _sb_train_df.groupby('week'):
    if len(_week_data) < 50 or _week_data['won_sb'].sum() < 3:
        continue  # insufficient signal for this week

    X_all = _week_data[_FEATURES].to_numpy()
    y_all = _week_data['won_sb'].to_numpy()

    # LOO: for each season in this week's data, train on the rest
    for _season in _week_data['season'].unique():
        mask_in   = _week_data['season'] != _season
        mask_held = _week_data['season'] == _season
        beta = _fit_logistic(X_all[mask_in.values], y_all[mask_in.values])

        held = _week_data[mask_held]
        X_held = held[_FEATURES].to_numpy()
        probs = _predict_logistic(X_held, beta)
        # Zero out mathematically eliminated teams, then renormalize.
        rid = int(held['ranking_id'].iloc[0])
        elim_set = _eliminated_cache.get(rid, set())
        if elim_set:
            for i, team in enumerate(held['team'].values):
                if team in elim_set:
                    probs[i] = 0.0
        total = probs.sum()
        if total > 0:
            probs = probs / total
        for team, p, rid_val in zip(held['team'].values, probs, held['ranking_id'].values):
            _sb_odds_cache[(int(rid_val), team)] = float(p)

    # Current/in-progress predictions: train on the full week dataset
    if not _sb_current_df.empty:
        cur_at_week = _sb_current_df[_sb_current_df['week'] == _week_val]
        if not cur_at_week.empty:
            beta_full = _fit_logistic(X_all, y_all)
            for _rid_val, rid_group in cur_at_week.groupby('ranking_id'):
                X_cur = rid_group[_FEATURES].to_numpy()
                probs = _predict_logistic(X_cur, beta_full)
                # Same elimination step
                elim_set = _eliminated_cache.get(int(_rid_val), set())
                if elim_set:
                    for i, team in enumerate(rid_group['team'].values):
                        if team in elim_set:
                            probs[i] = 0.0
                total = probs.sum()
                if total > 0:
                    probs = probs / total
                for team, p in zip(rid_group['team'].values, probs):
                    _sb_odds_cache[(int(_rid_val), team)] = float(p)

print(f"  RS SB-odds predictions cached for {len(_sb_odds_cache):,} (snapshot, team) pairs")


# ── Playoff SB odds (per-round LR + alive-set logic) ─────────────────────────
# Extends SB odds through the playoff snapshots (weeks 101-104). Teams who
# lost in earlier playoff rounds are explicitly zeroed out; alive teams
# get model predictions normalized so their probabilities sum to 100%.
#
# Alive sets are derived from games - playoff_teams = anyone who played in
# week 101+ that season; alive[week] = playoff_teams minus everyone who
# lost in any week up to and including this one. Works perfectly for
# completed seasons. For current in-progress at snap 101 (WC done, DR not
# played), bye teams won't appear in playoff_teams until DR - accepted
# limitation, fixed once DR games are scrape-able.

print("Computing playoff SB odds (rounds 101-103 + post-SB 104)...")
PLAYOFF_WEEKS = [101, 102, 103, 104]
PLAYOFF_TRAIN_WEEKS = [101, 102, 103]  # 104 is direct assignment from SB winner

def _playoff_state(season_games_df):
    pg = season_games_df[season_games_df['week'].isin(PLAYOFF_WEEKS)]
    if pg.empty:
        return {'playoff_teams': set(), 'alive_by_week': {}}
    playoff_teams = set(pg['home_team_name']) | set(pg['visitor_team_name'])
    eliminated = set()
    alive_by_week = {}
    for week in sorted({int(w) for w in pg['week'].unique() if int(w) in PLAYOFF_WEEKS}):
        for _, g in pg[pg['week'] == week].iterrows():
            if g['home_pts'] > g['visitor_pts']:
                eliminated.add(g['visitor_team_name'])
            elif g['visitor_pts'] > g['home_pts']:
                eliminated.add(g['home_team_name'])
        alive_by_week[week] = playoff_teams - eliminated
    return {'playoff_teams': playoff_teams, 'alive_by_week': alive_by_week}

_season_playoff_state = {int(s): _playoff_state(sub) for s, sub in games.groupby('season')}

# Build playoff training rows - alive teams at each playoff snapshot
_playoff_train_rows = []
for _, _r in df[df['week'].isin(PLAYOFF_TRAIN_WEEKS)].iterrows():
    _season = int(_r['season'])
    if _season not in _sb_winners:
        continue
    state = _season_playoff_state.get(_season, {}).get('alive_by_week', {})
    alive_at_week = state.get(int(_r['week']), set())
    if _r['name'] not in alive_at_week:
        continue
    if pd.isna(_r.get('rating_o')) or pd.isna(_r.get('rating_d')):
        continue
    _hfa = _hfa_val(_snapshot_hfa_cache.get(int(_r['ranking_id']), {}), _r['name'])
    if _hfa is None:
        continue
    _playoff_train_rows.append({
        'season':     _season,
        'week':       int(_r['week']),
        'ranking_id': int(_r['ranking_id']),
        'team':       _r['name'],
        'rating_o':   float(_r['rating_o']),
        'rating_d':   float(_r['rating_d']),
        'hfa':        float(_hfa),
        'won_sb':     1 if _r['name'] == _sb_winners[_season] else 0,
    })
_playoff_train_df = pd.DataFrame(_playoff_train_rows)
print(f"  Playoff training rows (alive teams only): {len(_playoff_train_df):,}")


def _zero_eliminated(rid, season, week_val):
    """Mark eliminated (in playoff field but not alive at this week) as 0%."""
    state = _season_playoff_state.get(season, {})
    playoff_teams = state.get('playoff_teams', set())
    alive_at_week = state.get('alive_by_week', {}).get(week_val, set())
    for team in playoff_teams - alive_at_week:
        _sb_odds_cache[(rid, team)] = 0.0


for _week_val, _week_data in _playoff_train_df.groupby('week'):
    if len(_week_data) < 30 or _week_data['won_sb'].sum() < 3:
        continue
    X_all = _week_data[_FEATURES].to_numpy()
    y_all = _week_data['won_sb'].to_numpy()

    # LOO for each historical season at this playoff week
    for _season in _week_data['season'].unique():
        mask_in = _week_data['season'] != _season
        mask_held = _week_data['season'] == _season
        beta = _fit_logistic(X_all[mask_in.values], y_all[mask_in.values])

        held = _week_data[mask_held]
        X_held = held[_FEATURES].to_numpy()
        probs = _predict_logistic(X_held, beta)
        total = probs.sum()
        if total > 0:
            probs = probs / total
        rid = int(held['ranking_id'].iloc[0])
        for team, p in zip(held['team'].values, probs):
            _sb_odds_cache[(rid, team)] = float(p)
        _zero_eliminated(rid, int(_season), int(_week_val))

    # Current/in-progress predictions: train on full history, predict for
    # current-season's alive teams at this playoff week (if any).
    if _sb_current_df is not None and not _sb_current_df.empty:
        pass  # handled below in a separate loop using game-aware alive set

# Current/in-progress playoff predictions: identify alive teams from
# game results, predict using full-history model.
_current_playoff_seasons = sorted(set(int(s) for s, _ in df.groupby('season')
                                     if int(s) not in _sb_winners))
for _season in _current_playoff_seasons:
    state = _season_playoff_state.get(_season, {})
    if not state.get('playoff_teams'):
        continue
    for _week_val in PLAYOFF_TRAIN_WEEKS:
        alive_at_week = state.get('alive_by_week', {}).get(_week_val, set())
        if not alive_at_week:
            continue
        # Find snapshot for this (season, week)
        snap_df = df[(df['season'] == _season) & (df['week'] == _week_val)]
        if snap_df.empty:
            continue
        rid = int(snap_df['ranking_id'].iloc[0])
        # Build feature matrix for alive teams from snap_df
        feature_rows = []
        feature_teams = []
        for _, _r in snap_df.iterrows():
            if _r['name'] not in alive_at_week:
                continue
            if pd.isna(_r.get('rating_o')) or pd.isna(_r.get('rating_d')):
                continue
            _hfa = _hfa_val(_snapshot_hfa_cache.get(rid, {}), _r['name'])
            if _hfa is None:
                continue
            feature_rows.append([float(_r['rating_o']), float(_r['rating_d']), float(_hfa)])
            feature_teams.append(_r['name'])
        if not feature_rows:
            continue
        # Fit on full historical playoff data at this week, predict for current alive teams
        train_at_week = _playoff_train_df[_playoff_train_df['week'] == _week_val]
        if len(train_at_week) < 30 or train_at_week['won_sb'].sum() < 3:
            continue
        beta = _fit_logistic(
            train_at_week[_FEATURES].to_numpy(),
            train_at_week['won_sb'].to_numpy(),
        )
        X_cur = np.array(feature_rows)
        probs = _predict_logistic(X_cur, beta)
        total = probs.sum()
        if total > 0:
            probs = probs / total
        for team, p in zip(feature_teams, probs):
            _sb_odds_cache[(rid, team)] = float(p)
        _zero_eliminated(rid, _season, _week_val)


# Snapshot 104 (post-SB): direct assignment - SB winner gets 100%, every
# other team that made the playoffs gets 0%, non-playoff teams stay absent.
for _season, _winner in _sb_winners.items():
    wk104 = df[(df['season'] == _season) & (df['week'] == 104)]
    if wk104.empty:
        continue
    rid = int(wk104['ranking_id'].iloc[0])
    playoff_teams = _season_playoff_state.get(_season, {}).get('playoff_teams', set())
    for team in playoff_teams:
        _sb_odds_cache[(rid, team)] = 1.0 if team == _winner else 0.0


# EOR playoff-field lock-in: at end of regular season we know EXACTLY who's
# in the playoffs. For COMPLETED seasons that's trivial - read week 101+
# games. For CURRENT in-progress at EOR (RS done, WC not played yet), we
# derive the field directly from RS standings using division winners + top
# wild cards by record. Either way, non-playoff teams get zeroed and the
# remaining playoff field renormalizes to 100%.

def _all_pairs_played(season, teams, rs_games):
    """True iff every pair of teams in the group played at least one game
    against each other this season. Required for NFL's h2h-sweep rule."""
    s = int(season)
    season_games = rs_games[rs_games['season'] == s]
    for i, t1 in enumerate(teams):
        for t2 in teams[i + 1:]:
            played = season_games[
                ((season_games['home_team_name'] == t1) & (season_games['visitor_team_name'] == t2))
                | ((season_games['home_team_name'] == t2) & (season_games['visitor_team_name'] == t1))
            ]
            if played.empty:
                return False
    return True


def _pick_top_wild_card(season, pool, rs_games):
    """Pick the single top team from a tied pool using the NFL wild-card
    tiebreaker cascade. Each step splits the surviving group; if a step
    cannot apply (missing data), continue to the next."""
    s_int = int(season)
    if len(pool) == 1:
        return pool[0]

    # Step 1: cull to one rep per division (using division tiebreaker).
    by_div = collections.defaultdict(list)
    for t in pool:
        by_div[div_for_season(t, s_int)].append(t)
    reps = []
    for teams in by_div.values():
        reps.append(teams[0] if len(teams) == 1
                    else _resolve_division_tie(s_int, teams, rs_games))
    if len(reps) == 1:
        return reps[0]

    def _filter(survivors, pcts):
        if not pcts or any(v is None for v in pcts.values()):
            return survivors
        best = max(pcts.values())
        winners = [t for t in survivors if pcts[t] == best]
        return winners if winners else survivors

    survivors = reps

    # Step 2: head-to-head sweep - only applicable when ALL teams in the
    # group played each other. Otherwise the rule is explicitly skipped
    # (NFL: "applicable only if one team beat all others or lost to all").
    if _all_pairs_played(s_int, survivors, rs_games):
        h2h = {t: _h2h_pct(s_int, t, [x for x in survivors if x != t], rs_games) for t in survivors}
        survivors = _filter(survivors, h2h)
        if len(survivors) == 1: return survivors[0]

    # Step 3: best conference record
    cr = {t: _conf_record_pct(s_int, t, rs_games) for t in survivors}
    survivors = _filter(survivors, cr)
    if len(survivors) == 1: return survivors[0]

    # Step 4: common games (>=4 common opponents)
    common = _common_opponents(s_int, survivors, rs_games)
    if common:
        cg = {t: _common_games_pct(s_int, t, common, rs_games) for t in survivors}
        survivors = _filter(survivors, cg)
        if len(survivors) == 1: return survivors[0]

    # Steps 5-6: strength of victory, strength of schedule
    season_records = _season_records_cache.get(s_int)
    if season_records is None:
        season_records = _build_season_records(s_int, rs_games)
        _season_records_cache[s_int] = season_records

    sov = {t: _strength_of_victory(s_int, t, season_records, rs_games) for t in survivors}
    survivors = _filter(survivors, sov)
    if len(survivors) == 1: return survivors[0]

    sos = {t: _strength_of_schedule(s_int, t, season_records, rs_games) for t in survivors}
    survivors = _filter(survivors, sos)
    if len(survivors) == 1: return survivors[0]

    # Steps 7-10: points-based cascade (combined ranking, net points)
    return _apply_points_based_cascade(s_int, survivors, rs_games)


def _resolve_wild_card_tie(season, tied_teams, num_to_pick, rs_games):
    """Pick `num_to_pick` teams from a tied pool, one at a time. Each pick
    redoes the full cull-and-cascade against the remaining pool - that's
    the actual NFL behavior, otherwise multiple non-div-winners from the
    same division can never both make the playoffs."""
    if len(tied_teams) <= num_to_pick:
        return list(tied_teams)
    picked = []
    pool = list(tied_teams)
    while len(picked) < num_to_pick and pool:
        top = _pick_top_wild_card(season, pool, rs_games)
        picked.append(top)
        pool.remove(top)
    return picked


def _playoff_field_from_eor_standings(season):
    """Derive the playoff field from end-of-RS standings using division
    winners + the NFL wild-card tiebreaker cascade for the remaining slots."""
    eor_rows = df[(df['season'] == season) & (df['last_week_of_regular_season'] == 1)]
    if eor_rows.empty:
        return set()

    seeds = _playoff_seeds_per_conf(season)
    div_winners_this_season = {t for (s, t) in _division_winners if s == season}

    # Build per-team info with raw W-L for proper tiebreaker-friendly pct.
    teams_info = []
    for _, r in eor_rows.iterrows():
        w, l, t = _parse_wlt(r.get('record', ''))
        eff_w = w + 0.5 * t
        n_games = w + l + t
        pct = (eff_w / n_games) if n_games > 0 else 0.0
        teams_info.append({
            'team': r['name'],
            'conf': conf_for_season(r['name'], season),
            'wins': eff_w,
            'pct':  pct,
        })

    playoff_field = set()
    for conf in {t['conf'] for t in teams_info}:
        conf_teams = [t for t in teams_info if t['conf'] == conf]
        conf_div_winners = [t for t in conf_teams if t['team'] in div_winners_this_season]
        playoff_field.update(t['team'] for t in conf_div_winners)

        remaining = seeds - len(conf_div_winners)
        if remaining <= 0:
            continue

        # Non-div-winners sorted by win pct descending; walk down picking
        # groups, applying the wild-card cascade within each tied group.
        non_div = [t for t in conf_teams if t['team'] not in div_winners_this_season]
        non_div.sort(key=lambda x: -x['pct'])

        i = 0
        slots_left = remaining
        while slots_left > 0 and i < len(non_div):
            current_pct = non_div[i]['pct']
            j = i
            while j < len(non_div) and non_div[j]['pct'] == current_pct:
                j += 1
            tied = [non_div[k]['team'] for k in range(i, j)]
            if len(tied) <= slots_left:
                playoff_field.update(tied)
                slots_left -= len(tied)
            else:
                picked = _resolve_wild_card_tie(season, tied, slots_left, _rs_games)
                playoff_field.update(picked)
                slots_left = 0
            i = j

    return playoff_field


def _resolve_playoff_field(season):
    """Prefer the games-based field for completed seasons (100% accurate);
    fall back to EOR standings derivation when no playoff games exist yet."""
    from_games = _season_playoff_state.get(season, {}).get('playoff_teams', set())
    if from_games:
        return from_games
    return _playoff_field_from_eor_standings(season)


# DEBUG: validate the standings-based derivation against the games-based ground
# truth across every completed season. Helps catch tiebreaker logic drift.
if os.environ.get('VALIDATE_PLAYOFF_FIELD'):
    print()
    print("Validating standings-derived playoff field against games-based truth...")
    n_perfect = 0
    n_mismatch = 0
    mismatch_seasons = []
    for _season, _state in _season_playoff_state.items():
        truth_set = _state.get('playoff_teams', set())
        if not truth_set:
            continue
        derived = _playoff_field_from_eor_standings(_season)
        missing = truth_set - derived
        extra = derived - truth_set
        if not missing and not extra:
            n_perfect += 1
        else:
            n_mismatch += 1
            mismatch_seasons.append((_season, sorted(missing), sorted(extra)))
    print(f"  Perfect match: {n_perfect}")
    print(f"  Mismatches: {n_mismatch}")
    for season, missing, extra in mismatch_seasons:
        print(f"    {season}: missing={missing}  extra={extra}")
    print()


print("  Locking in EOR playoff field per season...")
_eor_locked = 0
for _season in sorted(set(int(s) for s, _ in df.groupby('season'))):
    playoff_teams = _resolve_playoff_field(_season)
    if not playoff_teams:
        continue
    eor_rows = df[(df['season'] == _season) & (df['last_week_of_regular_season'] == 1)]
    if eor_rows.empty:
        continue
    eor_rid = int(eor_rows['ranking_id'].iloc[0])

    snap_probs = {}
    for team in eor_rows['name']:
        p = _sb_odds_cache.get((eor_rid, team))
        if p is not None:
            snap_probs[team] = p
    if not snap_probs:
        continue

    for team in list(snap_probs.keys()):
        if team not in playoff_teams:
            snap_probs[team] = 0.0

    total = sum(snap_probs.values())
    if total > 0:
        for team in snap_probs:
            snap_probs[team] = snap_probs[team] / total

    for team, p in snap_probs.items():
        _sb_odds_cache[(eor_rid, team)] = float(p)
    _eor_locked += 1
print(f"  EOR locked-in: {_eor_locked} seasons")


print(f"  Total SB-odds predictions cached: {len(_sb_odds_cache):,} (snapshot, team) pairs")


# Precompute per-snapshot SB-odds rank (1 = highest probability). Teams
# with 0% (eliminated) are unranked - display would just be "-". Built as
# a dict {ranking_id -> {team: rank}} so the per-team accessor is O(1).
_sb_odds_rank_cache = {}
_sb_pairs_by_rid = {}
for (rid, team), odds in _sb_odds_cache.items():
    if odds is None or odds <= 0:
        continue
    _sb_pairs_by_rid.setdefault(rid, []).append((team, odds))
for rid, pairs in _sb_pairs_by_rid.items():
    pairs.sort(key=lambda x: -x[1])
    rank_map = {}
    prev_odds = None
    prev_rank = 0
    for i, (team, odds) in enumerate(pairs, start=1):
        # Standard competition ranking: tied teams share the lower rank.
        if odds != prev_odds:
            prev_rank = i
            prev_odds = odds
        rank_map[team] = prev_rank
    _sb_odds_rank_cache[rid] = rank_map


def _sb_odds_val(ranking_id, team):
    """Return SB odds as float 0-1, or None if no prediction at this snapshot."""
    return _sb_odds_cache.get((int(ranking_id), team))


def _sb_odds_rk(ranking_id, team):
    """Return SB-odds rank (1 = best). None for eliminated (0%) teams."""
    rm = _sb_odds_rank_cache.get(int(ranking_id))
    return rm.get(team) if rm else None


# ── 1. Current standings ─────────────────────────────────────────────────────
print("Writing current_standings.json...")
latest_id = int(df['ranking_id'].max())
latest = df[df['ranking_id'] == latest_id].sort_values('rank').copy()
latest_date = str(latest['date'].iloc[0]) if not latest.empty else ''

standings_data = {
    'updated': latest_date,
    'hfa_window': _hfa_window_label,
    'teams': [
        {
            'rank':            int(r['rank']),
            'team':            r['name'],
            'display_name':    display_name(r['name'], r['season']),
            'conference':      conf_for_season(r['name'], r['season']),
            'division':        div_for_season(r['name'], r['season']),
            'division_winner': 1 if (int(r['season']), r['name']) in _division_winners else 0,
            'rating':          round(float(r['rating']), 3),
            'rating_o':        round(float(r['rating_o']), 3) if 'rating_o' in r and not pd.isna(r['rating_o']) else None,
            'rating_d':        round(float(r['rating_d']), 3) if 'rating_d' in r and not pd.isna(r['rating_d']) else None,
            'rank_o':          int(r['rank_o']) if 'rank_o' in r and not pd.isna(r['rank_o']) else None,
            'rank_d':          int(r['rank_d']) if 'rank_d' in r and not pd.isna(r['rank_d']) else None,
            'hfa':             _hfa_val(_hfa_lookup, r['name']),
            'hfa_rank':        _hfa_rk(_hfa_lookup, r['name']),
            'sb_odds':         _sb_odds_val(r['ranking_id'], r['name']),
            'sb_odds_rank':    _sb_odds_rk(r['ranking_id'], r['name']),
            'record':          clean(r['record']),
            'last_match':      era_aware_last_match(clean(r['lastgame']) if _played(r['lastgame']) else last_game_as_of(r['name'], r['season_week'], r['season']), r['season']),
            'sb_status':       int(r['sb_status']) if not pd.isna(r['sb_status']) else 0,
        }
        for _, r in latest.iterrows()
    ],
}
with open('docs/data/current_standings.json', 'w') as f:
    json.dump(standings_data, f, separators=(',', ':'))

# ── 2. GOAT tables (RS + PS) ─────────────────────────────────────────────────
# Two lists, mirroring GRIFFEY/SAKIC convention:
#   - goat_rs.json: top 50 by end-of-regular-season rating (any team).
#   - goat_ps.json: top 50 by end-of-postseason rating, restricted to teams
#     that reached the Super Bowl (sb_status >= 1) so the list shows actual
#     championship contenders, not playoff flameouts.
GOAT_TOP_N = 50
print("Writing goat_rs.json + goat_ps.json...")


# Short / disrupted seasons - flagged on GOAT/Champions/Standings/TeamSummary
# rows so the UI can tag them inline + footnote. 2020 NOT tagged: NFL played
# full 16-game schedule despite COVID disruption.
SHORT_SEASONS = {
    1982: {
        'tag': 'strike 9g',
        'category': 'labor',
        'note': 'The 1982 season was shortened to 9 games per team by a 57-day strike. Playoff format was expanded to 16 teams ("Super Bowl Tournament").',
    },
    1987: {
        'tag': 'strike 15g',
        'category': 'labor',
        'note': 'Three weeks of the 1987 season were played by replacement players ("scab games") during a 24-day strike; one additional week was cancelled outright, ending the season at 15 games per team.',
    },
}


def _build_goat(rows, sort_col='rating'):
    # Length rounds down to the nearest 10, capped at GOAT_TOP_N. The RS pool
    # (all teams) is huge so it always hits the cap (50); the champions-only PS
    # pool rounds down (e.g. 45 champion seasons -> top 40).
    _n = min(GOAT_TOP_N, (len(rows) // 10) * 10)
    rows = rows.sort_values(sort_col, ascending=False).head(_n).reset_index(drop=True)
    out = []
    for i, (_, r) in enumerate(rows.iterrows()):
        s = int(r['season'])
        reg = _reg_record_lookup.get((r['name'], s), '')
        # For RS snapshots, r['record'] equals the regular-season record
        # (no playoff games played yet), which would give playoff_record="0-0".
        # Use the team's FINAL record so the surfaced playoff_record reflects
        # their actual playoff outcome - matches GRIFFEY/SAKIC/COBI convention.
        final_record = _final_record_lookup.get((r['name'], s), r['record'])
        out.append({
            'rank':            i + 1,
            'team':            r['name'],
            'display_name':    display_name(r['name'], s),
            'conference':      conf_for_season(r['name'], s),
            'division':        div_for_season(r['name'], s),
            'division_winner': 1 if (s, r['name']) in _division_winners else 0,
            'season':          s,
            'short_season':          s in SHORT_SEASONS,
            'short_season_tag':      SHORT_SEASONS.get(s, {}).get('tag', '')      if s in SHORT_SEASONS else '',
            'short_season_category': SHORT_SEASONS.get(s, {}).get('category', '') if s in SHORT_SEASONS else '',
            'short_season_note':     SHORT_SEASONS.get(s, {}).get('note', '')     if s in SHORT_SEASONS else '',
            'rating':          round(float(r['rating']), 3),
            'rating_o':        round(float(r['rating_o']), 3) if 'rating_o' in r and not pd.isna(r['rating_o']) else None,
            'rating_d':        round(float(r['rating_d']), 3) if 'rating_d' in r and not pd.isna(r['rating_d']) else None,
            'record':          clean(r['record']),
            'regular_record':  reg,
            'playoff_record':  playoff_record(final_record, reg),
            'sb_status':       int(r['sb_status']) if not pd.isna(r['sb_status']) else 0,
        })
    return out


# Six GOAT files: {REACT, Offense, Defense} × {RS-end, PS-end}. The PS-end
# variants are restricted to Super Bowl participants (sb_status >= 1) so
# the list shows actual championship contenders, not playoff flameouts.
rs_rows = df[df['season_flag'] == 1].copy()
ps_rows = df[(df['season_flag'] == 2) & (df['sb_status'] == 2)].copy()  # champions only

goat_files = [
    ('goat_rs.json',   rs_rows, 'rating'),    # REACT, RS-end (canonical)
    ('goat_ps.json',   ps_rows, 'rating'),    # REACT, PS-end (canonical)
    ('goat_rs_o.json', rs_rows, 'rating_o'),  # Offense, RS-end
    ('goat_rs_d.json', rs_rows, 'rating_d'),  # Defense, RS-end
    ('goat_ps_o.json', ps_rows, 'rating_o'),  # Offense, PS-end
    ('goat_ps_d.json', ps_rows, 'rating_d'),  # Defense, PS-end
]
for fname, src, sort_col in goat_files:
    payload = _build_goat(src, sort_col=sort_col)
    with open(f'docs/data/{fname}', 'w') as f:
        json.dump(payload, f, separators=(',', ':'))

# ── 3. Per-team JSON files ───────────────────────────────────────────────────
print("Writing per-team JSON files...")
team_data = df[(df['is_game_day'] == 1) | (df['is_end_of_season'] == 1)].copy()
team_data = team_data.sort_values(['name', 'season', 'season_week'])

all_teams = sorted(df['name'].unique())
teams_index = []

for team in all_teams:
    tdf = team_data[team_data['name'] == team]
    if len(tdf) == 0:
        continue

    team_slug = slug(team)
    teams_index.append({
        'name': team,
        'display_name': current_display_name(team),
        'historical_names': historical_display_names(team),
        'conference': conf(team),
        'division': div(team),
        'slug': team_slug,
    })

    seasons = {}
    for season, sdf in tdf.groupby('season'):
        rs_end = _rs_end_sw.get(season)
        final_reg = _reg_record_lookup.get((team, int(season)))
        entries = []
        for _, r in sdf.sort_values('season_week').iterrows():
            in_postseason = (rs_end is not None) and (r['season_week'] > rs_end) and (final_reg is not None)
            if in_postseason:
                reg = final_reg
                po  = playoff_record(r['record'], final_reg)
            else:
                reg = clean(r['record'])
                po  = ''
            snap_hfa = _snapshot_hfa_cache.get(int(r['ranking_id']), {})
            entries.append({
                'date':              clean(r['date']),
                'season_week':       float(r['season_week']),
                'week':              int(r['week']),
                'week_label':        week_label(r['week']),
                'display_name':      display_name(team, season),
                'rating':            round(float(r['rating']), 3),
                'rank':              int(r['rank']),
                'rating_o':          round(float(r['rating_o']), 3) if 'rating_o' in r and not pd.isna(r['rating_o']) else None,
                'rating_d':          round(float(r['rating_d']), 3) if 'rating_d' in r and not pd.isna(r['rating_d']) else None,
                'rank_o':            int(r['rank_o']) if 'rank_o' in r and not pd.isna(r['rank_o']) else None,
                'rank_d':            int(r['rank_d']) if 'rank_d' in r and not pd.isna(r['rank_d']) else None,
                'hfa':               _hfa_val(snap_hfa, team),
                'hfa_rank':          _hfa_rk(snap_hfa, team),
                'sb_odds':           _sb_odds_val(r['ranking_id'], team),
                'sb_odds_rank':      _sb_odds_rk(r['ranking_id'], team),
                'record':            clean(r['record']),
                'regular_record':    reg,
                'playoff_record':    po,
                'last_match':        era_aware_last_match(clean(r['lastgame']) if _played(r['lastgame']) else last_game_as_of(team, r['season_week'], season), season),
                'is_end_of_season':  int(r['is_end_of_season']),
                'season_flag':       int(r['season_flag']),
                'is_playoff':        int(is_playoff(season, r['season_week'])),
                'sb_status':         int(r['sb_status']) if not pd.isna(r['sb_status']) else 0,
                'conference':        conf_for_season(team, season),
                'division':          div_for_season(team, season),
                'division_winner':   1 if (int(season), team) in _division_winners else 0,
            })
        seasons[int(season)] = entries

    with open(f'docs/data/teams/{team_slug}.json', 'w') as f:
        json.dump({
            'team': team,
            'conference': conf(team),
            'division': div(team),
            'seasons': seasons,
        }, f, separators=(',', ':'))

teams_index.sort(key=lambda x: x['name'])
with open('docs/data/teams_index.json', 'w') as f:
    json.dump(teams_index, f, separators=(',', ':'))

# ── 4. Season standings files ─────────────────────────────────────────────────
print("Writing season standings files...")
all_seasons = sorted(df['season'].unique())

for season in all_seasons:
    sdf = df[df['season'] == season]
    snapshots = []
    for ranking_id, rdf in sdf.groupby('ranking_id'):
        rdf = rdf.sort_values('rank')
        snap_sw   = rdf['season_week'].iloc[0]
        snap_date = clean(rdf['date'].iloc[0])
        wk        = int(rdf['week'].iloc[0])
        flag      = int(rdf['season_flag'].iloc[0])
        label     = snapshot_label(wk, flag)

        rs_end = _rs_end_sw.get(season)
        in_postseason = (rs_end is not None) and (snap_sw > rs_end)

        # Per-snapshot rolling HFA: 3-year window ending at this ranking_id.
        snap_hfa = _snapshot_hfa_cache.get(int(ranking_id), {})

        teams_snap = []
        for _, r in rdf.iterrows():
            if in_postseason:
                reg = _reg_record_lookup.get((r['name'], int(season)), r['record'])
                po  = playoff_record(r['record'], reg)
            else:
                reg = clean(r['record'])
                po  = ''
            played_today = _played(r['lastgame'])
            teams_snap.append({
                'rank':            int(r['rank']),
                'team':            r['name'],
                'display_name':    display_name(r['name'], season),
                'conference':      conf_for_season(r['name'], season),
                'division':        div_for_season(r['name'], season),
                'division_winner': 1 if (int(season), r['name']) in _division_winners else 0,
                'rating':          round(float(r['rating']), 3),
                'rating_o':        round(float(r['rating_o']), 3) if 'rating_o' in r and not pd.isna(r['rating_o']) else None,
                'rating_d':        round(float(r['rating_d']), 3) if 'rating_d' in r and not pd.isna(r['rating_d']) else None,
                'rank_o':          int(r['rank_o']) if 'rank_o' in r and not pd.isna(r['rank_o']) else None,
                'rank_d':          int(r['rank_d']) if 'rank_d' in r and not pd.isna(r['rank_d']) else None,
                'hfa':             _hfa_val(snap_hfa, r['name']),
                'hfa_rank':        _hfa_rk(snap_hfa, r['name']),
                'sb_odds':         _sb_odds_val(r['ranking_id'], r['name']),
                'sb_odds_rank':    _sb_odds_rk(r['ranking_id'], r['name']),
                'record':          clean(r['record']),
                'regular_record':  reg,
                'playoff_record':  po,
                'last_match':      era_aware_last_match(clean(r['lastgame']) if played_today else last_game_as_of(r['name'], snap_sw, season), season),
                'last_match_date': snap_date if played_today else last_game_date_as_of(r['name'], snap_sw, season),
                'sb_status':       int(r['sb_status']) if not pd.isna(r['sb_status']) else 0,
            })
        snapshots.append({
            'date':        snap_date,
            'season_week': float(snap_sw),
            'week':        wk,
            'label':       label,
            'teams':       teams_snap,
        })

    snapshots.sort(key=lambda x: x['season_week'])
    with open(f'docs/data/seasons/{int(season)}.json', 'w') as f:
        json.dump({'season': int(season), 'snapshots': snapshots}, f, separators=(',', ':'))

seasons_meta = {
    'seasons':    [int(s) for s in reversed(all_seasons)],
    'first_date': str(games['date'].min().date()),
    'last_date':  str(games['date'].max().date()),
    'generated_at': datetime.now(timezone.utc).isoformat(),
    'disrupted_seasons': {
        str(year): {'tag': info['tag'], 'category': info['category'], 'note': info['note']}
        for year, info in SHORT_SEASONS.items()
    },
}
with open('docs/data/seasons_index.json', 'w') as f:
    json.dump(seasons_meta, f, separators=(',', ':'))

# ── 5. Champions table (Super Bowl winners and runners-up) ───────────────────
print("Writing champions.json...")

# End-of-regular-season Super Bowl odds per (team, season): the model's title
# probability for that team at the End-of-regular-season snapshot (season_flag == 1,
# robust to the RS length changing over the eras - 14/16/17 games), i.e. how likely
# the eventual champion looked going into the playoffs. Powers the "Biggest favorites"
# / "Longest shots" lists. sb_odds_rank is the team's rank in those league-wide
# (sum-to-100%) odds at that snapshot.
_rs_sb_odds_lookup = {}
_rs_sb_odds_rank_lookup = {}
for _, row in df[df['season_flag'] == 1].iterrows():
    key = (row['name'], int(row['season']))
    _rs_sb_odds_lookup[key]      = _sb_odds_val(row['ranking_id'], row['name'])
    _rs_sb_odds_rank_lookup[key] = _sb_odds_rk(row['ranking_id'], row['name'])

champions = []

# Pre-SB snapshot lookup: week 103 is universally the snapshot taken AFTER the
# conference championship and BEFORE the Super Bowl, across all rated seasons
# (1971+). Used to evaluate matchup quality / closeness / upsets without the
# circularity of letting the SB result itself colour the "going-in" rating.
pre_sb_cols = ['name', 'season', 'rating', 'rank', 'record']
# Carry O/D through pre-SB lookup when available (defensive in case of older
# rebuilds without the columns - the engine commit 534caff is when they began).
for c in ('rating_o', 'rating_d', 'rank_o', 'rank_d'):
    if c in df.columns:
        pre_sb_cols.append(c)
pre_sb_df = df[df['week'] == 103][pre_sb_cols].copy()
pre_sb_lookup = {(r['name'], int(r['season'])): r for _, r in pre_sb_df.iterrows()}

def pre_sb_fields(name, season, reg_record):
    """Return the pre-SB rating/rank/playoff_record block, or empty if missing."""
    p = pre_sb_lookup.get((name, int(season)))
    if p is None:
        return {}
    out = {
        'rating_pre':         round(float(p['rating']),  3),
        'rank_pre':           int(p['rank']),
        'playoff_record_pre': playoff_record(clean(p['record']), reg_record),
    }
    if 'rating_o' in p and not pd.isna(p['rating_o']):
        out['rating_o_pre'] = round(float(p['rating_o']), 3)
        out['rank_o_pre']   = int(p['rank_o']) if not pd.isna(p['rank_o']) else None
    if 'rating_d' in p and not pd.isna(p['rating_d']):
        out['rating_d_pre'] = round(float(p['rating_d']), 3)
        out['rank_d_pre']   = int(p['rank_d']) if not pd.isna(p['rank_d']) else None
    return out

for season in sorted(df['season'].unique(), reverse=True):
    sdf = df[(df['season'] == season) & (df['season_flag'] == 2)]
    if sdf.empty:
        continue
    champ_row = sdf[sdf['sb_champ'] == 1]
    ru_row = sdf[sdf['sb_runnerup'] == 1]
    if champ_row.empty or ru_row.empty:
        continue

    cr = champ_row.iloc[0]
    rr = ru_row.iloc[0]

    # Final score: the Super Bowl game itself (week=104)
    sb_game = games[(games['season'] == season) & (games['week'] == 104)]
    final_score = ''
    if not sb_game.empty:
        g = sb_game.iloc[0]
        # winner column always has the SB winner; format as "winner_pts-loser_pts"
        final_score = f"{int(g['ptsw'])}-{int(g['ptsl'])}"

    champ_reg = _reg_record_lookup.get((cr['name'], int(season)), '')
    ru_reg    = _reg_record_lookup.get((rr['name'], int(season)), '')

    champions.append({
        'season':       int(season),
        'final_score':  final_score,
        # NFL is single-elimination - no series, only the Super Bowl is one game.
        'champion': {
            'team':           cr['name'],
            'display_name':   display_name(cr['name'], season),
            'conference':     conf_for_season(cr['name'], season),
            'division':        div_for_season(cr['name'], season),
            'division_winner': 1 if (int(season), cr['name']) in _division_winners else 0,
            'rating':         round(float(cr['rating']), 3),
            'rating_rs':      _rs_rating_lookup.get((cr['name'], int(season))),
            'rs_sb_odds':      _rs_sb_odds_lookup.get((cr['name'], int(season))),
            'rs_sb_odds_rank': _rs_sb_odds_rank_lookup.get((cr['name'], int(season))),
            'rank':           int(cr['rank']),
            'rating_o':       round(float(cr['rating_o']), 3) if 'rating_o' in cr and not pd.isna(cr['rating_o']) else None,
            'rating_d':       round(float(cr['rating_d']), 3) if 'rating_d' in cr and not pd.isna(cr['rating_d']) else None,
            'rank_o':         int(cr['rank_o']) if 'rank_o' in cr and not pd.isna(cr['rank_o']) else None,
            'rank_d':         int(cr['rank_d']) if 'rank_d' in cr and not pd.isna(cr['rank_d']) else None,
            'record':         clean(cr['record']),
            'regular_record': champ_reg,
            'playoff_record': playoff_record(cr['record'], champ_reg),
            **pre_sb_fields(cr['name'], season, champ_reg),
        },
        'runner_up': {
            'team':           rr['name'],
            'display_name':   display_name(rr['name'], season),
            'conference':     conf_for_season(rr['name'], season),
            'division':        div_for_season(rr['name'], season),
            'division_winner': 1 if (int(season), rr['name']) in _division_winners else 0,
            'rating':         round(float(rr['rating']), 3),
            'rank':           int(rr['rank']),
            'rating_o':       round(float(rr['rating_o']), 3) if 'rating_o' in rr and not pd.isna(rr['rating_o']) else None,
            'rating_d':       round(float(rr['rating_d']), 3) if 'rating_d' in rr and not pd.isna(rr['rating_d']) else None,
            'rank_o':         int(rr['rank_o']) if 'rank_o' in rr and not pd.isna(rr['rank_o']) else None,
            'rank_d':         int(rr['rank_d']) if 'rank_d' in rr and not pd.isna(rr['rank_d']) else None,
            'record':         clean(rr['record']),
            'regular_record': ru_reg,
            'playoff_record': playoff_record(rr['record'], ru_reg),
            **pre_sb_fields(rr['name'], season, ru_reg),
        },
    })

# Pre-rated-data Super Bowl titles (SB I through V).
# Our rating data starts at season 1971 due to the 20-week warm-up, so SBs from
# the 1966-1970 seasons aren't otherwise captured. Counts only SB-era titles -
# pre-Super Bowl NFL championships (1920-1965) and AFL championships are NOT included.
PRE_RATED_CHAMPIONSHIPS = {
    'Green Bay Packers':  2,  # SB I (1966), SB II (1967)
    'New York Jets':      1,  # SB III (1968)
    'Kansas City Chiefs': 1,  # SB IV (1969)
    'Baltimore Colts':    1,  # SB V (1970)
}

PRE_RATED_RUNNER_UPS = {
    'Kansas City Chiefs': 1,  # SB I (1966)
    'Oakland Raiders':    1,  # SB II (1967)
    'Baltimore Colts':    1,  # SB III (1968)
    'Minnesota Vikings':  1,  # SB IV (1969)
    'Dallas Cowboys':     1,  # SB V (1970)
}

# Running counts, seeded with pre-rated totals
_champ_count = dict(PRE_RATED_CHAMPIONSHIPS)
_ru_count    = dict(PRE_RATED_RUNNER_UPS)
for entry in reversed(champions):
    ct = entry['champion']['team']
    rt = entry['runner_up']['team']
    _champ_count[ct] = _champ_count.get(ct, 0) + 1
    _ru_count[rt]    = _ru_count.get(rt, 0) + 1
    entry['champion']['title_count']      = _champ_count[ct]
    entry['runner_up']['runner_up_count'] = _ru_count[rt]

# Pre-rated Super Bowls (I-V, seasons 1966-1970): listed for completeness on
# the Super Bowls tab. No ratings/ranks (data anchor is 1971); team cells aren't
# linked. Counts here are the running totals AT THAT POINT IN TIME, independent
# of the seeded counts above (which feed downstream rated rows).
PRE_RATED_SB_ROWS = [
    # SB V - Jan 1971 - season 1970 - first post-merger Super Bowl, no AFL
    {'sb_num': 'V', 'season': 1970, 'final_score': '16-13',
     'champion':  {'team': 'Baltimore Colts',    'title_count': 1, 'regular_record': '11-2-1', 'playoff_record': '3-0'},
     'runner_up': {'team': 'Dallas Cowboys',     'runner_up_count': 1, 'regular_record': '10-4', 'playoff_record': '2-1'}},
    # SB IV - Jan 1970 - season 1969 - last AFL season
    {'sb_num': 'IV', 'season': 1969, 'final_score': '23-7',
     'champion':  {'team': 'Kansas City Chiefs', 'title_count': 1, 'regular_record': '11-3', 'playoff_record': '3-0', 'afl': True},
     'runner_up': {'team': 'Minnesota Vikings',  'runner_up_count': 1, 'regular_record': '12-2', 'playoff_record': '2-1'}},
    # SB III - Jan 1969 - season 1968
    {'sb_num': 'III', 'season': 1968, 'final_score': '16-7',
     'champion':  {'team': 'New York Jets',      'title_count': 1, 'regular_record': '11-3', 'playoff_record': '2-0', 'afl': True},
     'runner_up': {'team': 'Baltimore Colts',    'runner_up_count': 1, 'regular_record': '13-1', 'playoff_record': '2-1'}},
    # SB II - Jan 1968 - season 1967
    {'sb_num': 'II', 'season': 1967, 'final_score': '33-14',
     'champion':  {'team': 'Green Bay Packers',  'title_count': 2, 'regular_record': '9-4-1', 'playoff_record': '3-0'},
     'runner_up': {'team': 'Oakland Raiders',    'runner_up_count': 1, 'regular_record': '13-1', 'playoff_record': '1-1', 'afl': True}},
    # SB I - Jan 1967 - season 1966
    {'sb_num': 'I', 'season': 1966, 'final_score': '35-10',
     'champion':  {'team': 'Green Bay Packers',  'title_count': 1, 'regular_record': '12-2',   'playoff_record': '2-0'},
     'runner_up': {'team': 'Kansas City Chiefs', 'runner_up_count': 1, 'regular_record': '11-2-1', 'playoff_record': '1-1', 'afl': True}},
]

for row in PRE_RATED_SB_ROWS:
    champ_entry = {
        'team':           row['champion']['team'],
        'display_name':   row['champion']['team'],
        'title_count':    row['champion']['title_count'],
        'regular_record': row['champion']['regular_record'],
        'playoff_record': row['champion']['playoff_record'],
    }
    if row['champion'].get('afl'):
        champ_entry['afl'] = True
    ru_entry = {
        'team':            row['runner_up']['team'],
        'display_name':    row['runner_up']['team'],
        'runner_up_count': row['runner_up']['runner_up_count'],
        'regular_record':  row['runner_up']['regular_record'],
        'playoff_record':  row['runner_up']['playoff_record'],
    }
    if row['runner_up'].get('afl'):
        ru_entry['afl'] = True
    champions.append({
        'season':      row['season'],
        'final_score': row['final_score'],
        'pre_rated':   True,
        'champion':    champ_entry,
        'runner_up':   ru_entry,
    })

with open('docs/data/champions.json', 'w') as f:
    json.dump({'NFL': champions}, f, separators=(',', ':'))

print(f"Done. {len(teams_index)} teams, {len(standings_data['teams'])} in current standings.")
print(f"Wrote {len(all_seasons)} season files. Standings date: {latest_date}")

# Hygiene: flag any rated team missing from TEAM_CONFERENCE. Without this,
# expansion teams (or future renames) silently fall through to ('Other',
# 'Other') and disappear from the conference filter pillbox.
_unknown = sorted({t for t in df['name'].unique() if t not in TEAM_CONFERENCE})
if _unknown:
    print()
    print('⚠️  WARNING: teams in rated data missing from TEAM_CONFERENCE:')
    for t in _unknown:
        print(f'    - {t!r}')
    print('    These teams will display as "Other" until added.')
    print()
