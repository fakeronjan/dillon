"""
generate_data.py — reads dillon_ratings_with_standings.csv and writes JSON for the DILLON web frontend.
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
import json
import os
import re
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
    # AFC East — stable from 1970
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

    # AFC South (created 2002) — Houston/Tennessee lineage was AFC Central
    'Houston Oilers':           [(1970, 1996, 'AFC', 'Central')],
    'Tennessee Oilers':         [(1997, 1998, 'AFC', 'Central')],
    'Tennessee Titans':         [(1999, 2001, 'AFC', 'Central'),
                                 (2002, 9999, 'AFC', 'South')],
    'Jacksonville Jaguars':     [(1995, 2001, 'AFC', 'Central'),
                                 (2002, 9999, 'AFC', 'South')],
    'Houston Texans':           [(2002, 9999, 'AFC', 'South')],

    # AFC West — stable, with relocations within division
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

    # NFC East — stable from 1970
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
    'Arizona Cardinals':        [(1994, 2001, 'NFC', 'East'),
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

    # NFC South (created 2002) — Falcons/Saints were NFC West, Bucs NFC Central
    'Atlanta Falcons':          [(1970, 2001, 'NFC', 'West'),
                                 (2002, 9999, 'NFC', 'South')],
    'Carolina Panthers':        [(1995, 2001, 'NFC', 'West'),
                                 (2002, 9999, 'NFC', 'South')],
    'New Orleans Saints':       [(1970, 2001, 'NFC', 'West'),
                                 (2002, 9999, 'NFC', 'South')],

    # NFC West — Rams + 49ers (and Seahawks 1976, Bucs/etc handled above)
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
    — both must be treated as "didn't play" or the forward-fill of
    last_match breaks for any week a team had a bye."""
    if result is None or pd.isna(result):
        return False
    s = str(result).strip()
    return s not in ('', 'Bye / No Game')


# is_game_day: row where the team actually played that week
df['is_game_day'] = df['lastgame'].apply(_played).astype(int)
df['is_end_of_season'] = df['season_flag'].isin([1, 2]).astype(int)

# Per-(team, season) forward-filled last game. Keying by season prevents
# cross-season carry-forward — at the start of a new season, teams that
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

# Final (post-playoff) record per (team, season) from season_flag == 2 snapshots.
# Used so the GOAT-RS view can show each team's actual playoff outcome alongside
# their regular-season record — mirrors GRIFFEY/SAKIC/COBI, where playoff_record
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
    """Break a division tie using NFL-style tiebreakers (head-to-head, then
    division record, then conference record). Alphabetical name as final
    fallback. Returns the winning team name."""
    if len(tied_teams) == 1:
        return tied_teams[0]
    s_int = int(season)
    # 1) Head-to-head among tied teams
    h2h = {t: _h2h_pct(s_int, t, [x for x in tied_teams if x != t], rs_games) for t in tied_teams}
    if all(v is not None for v in h2h.values()):
        best = max(h2h.values())
        survivors = [t for t in tied_teams if h2h[t] == best]
        if len(survivors) == 1:
            return survivors[0]
        tied_teams = survivors
    # 2) Best record in division games
    drec = {t: _div_record_pct(s_int, t, rs_games) for t in tied_teams}
    if all(v is not None for v in drec.values()):
        best = max(drec.values())
        survivors = [t for t in tied_teams if drec[t] == best]
        if len(survivors) == 1:
            return survivors[0]
        tied_teams = survivors
    # 3) Best record in conference games
    crec = {t: _conf_record_pct(s_int, t, rs_games) for t in tied_teams}
    if all(v is not None for v in crec.values()):
        best = max(crec.values())
        survivors = [t for t in tied_teams if crec[t] == best]
        if len(survivors) == 1:
            return survivors[0]
        tied_teams = survivors
    # 4) Final fallback: alphabetical
    return sorted(tied_teams)[0]


# ── Division winners (per season + (conference, division)) ───────────────────
# Tag the team with the best RS record in each (conference, division) at end
# of regular season. Real NFL tiebreakers are 9 steps deep — we apply the first
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


# ── Team-specific home field advantage (3-year rolling window) ───────────────
# For each team T and each snapshot R, compute their home margin premium over
# what their rating gap alone would predict, using the 3 NFL seasons ending at
# R (with R's season contributing only games played before R within the season):
#
#   h_T(R) = mean over T's home games in window of:
#              (home_pts - away_pts) - (T_rating - opp_rating)
#
# Where ratings are the team's rating GOING INTO the game (prior snapshot).
# Excludes neutral-site games. League-wide mean lands near the global 2.5
# constant; altitude / cold-weather / loud-dome teams sit above, soft-market
# teams below. 3-year window strikes the best balance of stability and recency
# (per diagnostic review 2026-06-03; 1yr too noisy, 5yr too era-blended).
#
# Per-snapshot rolling lets you see how a team's home advantage has evolved —
# Seahawks' 12th Man peak in 2013-14 vs the post-Wilson decline, Lambeau's
# steady decade, etc.

print("Computing per-snapshot home field advantage (3-year rolling window)...")
_HFA_WINDOW = 3
_MIN_HOME_GAMES = 10  # below this, hfa estimate is too noisy; suppress

_rsorted = df.sort_values(['name', 'ranking_id']).copy()
_rsorted['rating_prior'] = _rsorted.groupby('name')['rating'].shift(1)
_rating_prior_lookup = _rsorted.set_index(['name', 'season', 'week'])['rating_prior'].to_dict()

def _prior_rating(name, season, week):
    return _rating_prior_lookup.get((name, int(season), int(week)))

# Build the full hfa_contribution table — one row per non-neutral game with
# valid prior ratings for both teams. Reused across all snapshot lookups.
_all_hfa_games = games[games['is_neutral'] == 0].copy()
_all_hfa_games['home_rating'] = _all_hfa_games.apply(
    lambda g: _prior_rating(g['home_team_name'], g['season'], g['week']), axis=1)
_all_hfa_games['away_rating'] = _all_hfa_games.apply(
    lambda g: _prior_rating(g['visitor_team_name'], g['season'], g['week']), axis=1)
_all_hfa_games = _all_hfa_games.dropna(subset=['home_rating', 'away_rating']).copy()
_all_hfa_games['hfa_contribution'] = (
    (_all_hfa_games['home_pts'] - _all_hfa_games['visitor_pts'])
    - (_all_hfa_games['home_rating'] - _all_hfa_games['away_rating'])
)
print(f"  HFA-eligible games: {len(_all_hfa_games):,}")

def _hfa_snapshot(season, ranking_id):
    """Return {team: {'hfa': float, 'rank': int}} for the 3-year window ending
    at this snapshot. Window includes prior 2 seasons in full + games from the
    snapshot's season up to (and including) the snapshot's ranking_id. Rank is
    computed within this snapshot (1 = highest HFA in the league)."""
    window_seasons = list(range(int(season) - _HFA_WINDOW + 1, int(season) + 1))
    win = _all_hfa_games[
        _all_hfa_games['season'].isin(window_seasons)
        & (_all_hfa_games['cume_week_id'] <= ranking_id)
    ]
    if win.empty:
        return {}
    g = win.groupby('home_team_name')['hfa_contribution'].agg(['mean', 'size'])
    g = g[g['size'] >= _MIN_HOME_GAMES].copy()
    if g.empty:
        return {}
    g['hfa'] = g['mean'].round(2)
    g['rank'] = g['hfa'].rank(ascending=False, method='min').astype(int)
    return {team: {'hfa': float(row['hfa']), 'rank': int(row['rank'])}
            for team, row in g.iterrows()}

# Pre-compute snapshot HFA for every ranking_id so per-season + per-team
# writers share the cache. ~1100 snapshot computations; runs in seconds.
_LATEST_SEASON = int(df['season'].max())
_LATEST_RID = int(df['ranking_id'].max())
_snapshot_seasons = df.groupby('ranking_id')['season'].first().astype(int).to_dict()

_snapshot_hfa_cache = {}  # ranking_id -> {team: hfa}
for _rid in sorted(_snapshot_seasons):
    _snapshot_hfa_cache[int(_rid)] = _hfa_snapshot(_snapshot_seasons[_rid], int(_rid))

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

_hfa_window_label = f"{_LATEST_SEASON - _HFA_WINDOW + 1}-{_LATEST_SEASON}"
print(f"  Cached HFA for {len(_snapshot_hfa_cache):,} snapshots; "
      f"latest snapshot has HFA for {len(_hfa_lookup)} teams "
      f"(window {_hfa_window_label})")


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


# Short / disrupted seasons — flagged on GOAT/Champions/Standings/TeamSummary
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
    rows = rows.sort_values(sort_col, ascending=False).head(GOAT_TOP_N).reset_index(drop=True)
    out = []
    for i, (_, r) in enumerate(rows.iterrows()):
        s = int(r['season'])
        reg = _reg_record_lookup.get((r['name'], s), '')
        # For RS snapshots, r['record'] equals the regular-season record
        # (no playoff games played yet), which would give playoff_record="0-0".
        # Use the team's FINAL record so the surfaced playoff_record reflects
        # their actual playoff outcome — matches GRIFFEY/SAKIC/COBI convention.
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
ps_rows = df[(df['season_flag'] == 2) & (df['sb_status'] >= 1)].copy()

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

champions = []

# Pre-SB snapshot lookup: week 103 is universally the snapshot taken AFTER the
# conference championship and BEFORE the Super Bowl, across all rated seasons
# (1971+). Used to evaluate matchup quality / closeness / upsets without the
# circularity of letting the SB result itself colour the "going-in" rating.
pre_sb_cols = ['name', 'season', 'rating', 'rank', 'record']
# Carry O/D through pre-SB lookup when available (defensive in case of older
# rebuilds without the columns — the engine commit 534caff is when they began).
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
        # NFL is single-elimination — no series, only the Super Bowl is one game.
        'champion': {
            'team':           cr['name'],
            'display_name':   display_name(cr['name'], season),
            'conference':     conf_for_season(cr['name'], season),
            'division':        div_for_season(cr['name'], season),
            'division_winner': 1 if (int(season), cr['name']) in _division_winners else 0,
            'rating':         round(float(cr['rating']), 3),
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
# the 1966-1970 seasons aren't otherwise captured. Counts only SB-era titles —
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
    # SB V — Jan 1971 — season 1970 — first post-merger Super Bowl, no AFL
    {'sb_num': 'V', 'season': 1970, 'final_score': '16-13',
     'champion':  {'team': 'Baltimore Colts',    'title_count': 1, 'regular_record': '11-2-1', 'playoff_record': '3-0'},
     'runner_up': {'team': 'Dallas Cowboys',     'runner_up_count': 1, 'regular_record': '10-4', 'playoff_record': '2-1'}},
    # SB IV — Jan 1970 — season 1969 — last AFL season
    {'sb_num': 'IV', 'season': 1969, 'final_score': '23-7',
     'champion':  {'team': 'Kansas City Chiefs', 'title_count': 1, 'regular_record': '11-3', 'playoff_record': '3-0', 'afl': True},
     'runner_up': {'team': 'Minnesota Vikings',  'runner_up_count': 1, 'regular_record': '12-2', 'playoff_record': '2-1'}},
    # SB III — Jan 1969 — season 1968
    {'sb_num': 'III', 'season': 1968, 'final_score': '16-7',
     'champion':  {'team': 'New York Jets',      'title_count': 1, 'regular_record': '11-3', 'playoff_record': '2-0', 'afl': True},
     'runner_up': {'team': 'Baltimore Colts',    'runner_up_count': 1, 'regular_record': '13-1', 'playoff_record': '2-1'}},
    # SB II — Jan 1968 — season 1967
    {'sb_num': 'II', 'season': 1967, 'final_score': '33-14',
     'champion':  {'team': 'Green Bay Packers',  'title_count': 2, 'regular_record': '9-4-1', 'playoff_record': '3-0'},
     'runner_up': {'team': 'Oakland Raiders',    'runner_up_count': 1, 'regular_record': '13-1', 'playoff_record': '1-1', 'afl': True}},
    # SB I — Jan 1967 — season 1966
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
