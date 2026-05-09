"""
generate_data.py — reads dillon_ratings_with_standings.csv and writes JSON for the DILLON web frontend.
Run after dillon.py. Outputs to docs/data/.

NFL-specific tweaks vs LOBO/DUNCAN:
  - Two ratings per team per snapshot (REACT 20-week + HOTTAKE 10-week)
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


# ── Per-(team, season) conference overrides ──────────────────────────────────
# Most NFL franchises have stayed in their conference forever, but a handful
# of moves require season-aware lookup so historical Standings / Team Summary /
# Champions / GOAT entries don't show the wrong conference for past seasons.
#
# Format: (team_name, first_season, last_season, conference, division)
# - 1976+ realignment: Tampa Bay only spent its expansion year in AFC West
#   before swapping with Seattle to NFC Central.
# - 2002 realignment: Seahawks moved AFC West → NFC West when the league
#   reorganized into 8 four-team divisions.
# Cardinals NFC East → NFC West (2002) is a division-only change (always NFC)
# and doesn't need an override here. Browns AFC Central → AFC North (2002)
# similarly stays in AFC. Same for Steelers/Bengals/Ravens/Titans/Jaguars.
CONF_OVERRIDES = [
    ('Seattle Seahawks',      1976, 2001, 'AFC', 'West'),
    ('Tampa Bay Buccaneers',  1976, 1976, 'AFC', 'West'),
]


def _override_for(team, season):
    s = int(season)
    for (t, y0, y1, c, d) in CONF_OVERRIDES:
        if t == team and y0 <= s <= y1:
            return (c, d)
    return None


def conf_for_season(team, season):
    o = _override_for(team, season)
    return o[0] if o else conf(team)


def div_for_season(team, season):
    o = _override_for(team, season)
    return o[1] if o else div(team)


def clean(val):
    if pd.isna(val):
        return ''
    return str(val)


def slug(name):
    return re.sub(r'[^\w]', '_', name).strip('_')


# is_game_day: row where the team actually played that week
df['is_game_day'] = (df['lastgame'] != 'Bye / No Game').astype(int)
df['is_end_of_season'] = df['season_flag'].isin([1, 2]).astype(int)

# Per-team forward-filled last game (so EOS rows that aren't game days still show prior game)
_last_game_history = {}
for team, tdf in df[df['is_game_day'] == 1].sort_values('season_week').groupby('name'):
    _last_game_history[team] = (
        list(tdf['season_week']),
        list(tdf['lastgame']),
        list(tdf['date']),
    )


def last_game_as_of(team, sw):
    entry = _last_game_history.get(team)
    if not entry:
        return ''
    sws, games_list, _ = entry
    idx = bisect_right(sws, sw) - 1
    return games_list[idx] if idx >= 0 else ''


def last_game_date_as_of(team, sw):
    entry = _last_game_history.get(team)
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
        return f'{base} · End of season'
    return base


# ── 1. Current standings ─────────────────────────────────────────────────────
print("Writing current_standings.json...")
latest_id = int(df['ranking_id'].max())
latest = df[df['ranking_id'] == latest_id].sort_values('rank').copy()
latest_date = str(latest['date'].iloc[0]) if not latest.empty else ''

standings_data = {
    'updated': latest_date,
    'teams': [
        {
            'rank':            int(r['rank']),
            'rank2':           int(r['rank2']) if not pd.isna(r['rank2']) else None,
            'team':            r['name'],
            'conference':      conf_for_season(r['name'], r['season']),
            'division':        div_for_season(r['name'], r['season']),
            'rating':          round(float(r['rating']), 3),
            'rating2':         round(float(r['rating2']), 3) if not pd.isna(r['rating2']) else None,
            'record':          clean(r['record']),
            'last_match':      clean(r['lastgame']) if r['lastgame'] != 'Bye / No Game' else last_game_as_of(r['name'], r['season_week']),
            'sb_status':       int(r['sb_status']) if not pd.isna(r['sb_status']) else 0,
        }
        for _, r in latest.iterrows()
    ],
}
with open('docs/data/current_standings.json', 'w') as f:
    json.dump(standings_data, f, separators=(',', ':'))

# ── 2. GOAT table ─────────────────────────────────────────────────────────────
# Only fully-complete seasons (flag=2 = SB ended).
print("Writing goat_teams.json...")
eos_all = df[df['season_flag'] == 2].copy()
eos_top = eos_all.sort_values('rating', ascending=False).head(50).reset_index(drop=True)

goat_data = []
for i, (_, r) in enumerate(eos_top.iterrows()):
    reg = _reg_record_lookup.get((r['name'], int(r['season'])), '')
    goat_data.append({
        'rank':           i + 1,
        'team':           r['name'],
        'conference':     conf_for_season(r['name'], r['season']),
        'division':       div_for_season(r['name'], r['season']),
        'season':         int(r['season']),
        'rating':         round(float(r['rating']), 3),
        'rating2':        round(float(r['rating2']), 3) if not pd.isna(r['rating2']) else None,
        'rank2':          int(r['rank2']) if not pd.isna(r['rank2']) else None,
        'record':         clean(r['record']),
        'regular_record': reg,
        'playoff_record': playoff_record(r['record'], reg),
        'sb_status':      int(r['sb_status']) if not pd.isna(r['sb_status']) else 0,
    })
with open('docs/data/goat_teams.json', 'w') as f:
    json.dump(goat_data, f, separators=(',', ':'))

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
            entries.append({
                'date':              clean(r['date']),
                'season_week':       float(r['season_week']),
                'week':              int(r['week']),
                'week_label':        week_label(r['week']),
                'rating':            round(float(r['rating']), 3),
                'rank':              int(r['rank']),
                'rating2':           round(float(r['rating2']), 3) if not pd.isna(r['rating2']) else None,
                'rank2':             int(r['rank2']) if not pd.isna(r['rank2']) else None,
                'record':            clean(r['record']),
                'regular_record':    reg,
                'playoff_record':    po,
                'last_match':        clean(r['lastgame']) if r['lastgame'] != 'Bye / No Game' else last_game_as_of(team, r['season_week']),
                'is_end_of_season':  int(r['is_end_of_season']),
                'season_flag':       int(r['season_flag']),
                'is_playoff':        int(is_playoff(season, r['season_week'])),
                'sb_status':         int(r['sb_status']) if not pd.isna(r['sb_status']) else 0,
                'conference':        conf_for_season(team, season),
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

        teams_snap = []
        for _, r in rdf.iterrows():
            if in_postseason:
                reg = _reg_record_lookup.get((r['name'], int(season)), r['record'])
                po  = playoff_record(r['record'], reg)
            else:
                reg = clean(r['record'])
                po  = ''
            played_today = r['lastgame'] != 'Bye / No Game'
            teams_snap.append({
                'rank':            int(r['rank']),
                'rank2':           int(r['rank2']) if not pd.isna(r['rank2']) else None,
                'team':            r['name'],
                'conference':      conf_for_season(r['name'], season),
                'division':        div_for_season(r['name'], season),
                'rating':          round(float(r['rating']), 3),
                'rating2':         round(float(r['rating2']), 3) if not pd.isna(r['rating2']) else None,
                'record':          clean(r['record']),
                'regular_record':  reg,
                'playoff_record':  po,
                'last_match':      clean(r['lastgame']) if played_today else last_game_as_of(r['name'], snap_sw),
                'last_match_date': snap_date if played_today else last_game_date_as_of(r['name'], snap_sw),
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
}
with open('docs/data/seasons_index.json', 'w') as f:
    json.dump(seasons_meta, f, separators=(',', ':'))

# ── 5. Champions table (Super Bowl winners and runners-up) ───────────────────
print("Writing champions.json...")

champions = []
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
            'conference':     conf_for_season(cr['name'], season),
            'division':       div_for_season(cr['name'], season),
            'rating':         round(float(cr['rating']), 3),
            'rating2':        round(float(cr['rating2']), 3) if not pd.isna(cr['rating2']) else None,
            'rank':           int(cr['rank']),
            'rank2':          int(cr['rank2']) if not pd.isna(cr['rank2']) else None,
            'record':         clean(cr['record']),
            'regular_record': champ_reg,
            'playoff_record': playoff_record(cr['record'], champ_reg),
        },
        'runner_up': {
            'team':           rr['name'],
            'conference':     conf_for_season(rr['name'], season),
            'division':       div_for_season(rr['name'], season),
            'rating':         round(float(rr['rating']), 3),
            'rating2':        round(float(rr['rating2']), 3) if not pd.isna(rr['rating2']) else None,
            'rank':           int(rr['rank']),
            'rank2':          int(rr['rank2']) if not pd.isna(rr['rank2']) else None,
            'record':         clean(rr['record']),
            'regular_record': ru_reg,
            'playoff_record': playoff_record(rr['record'], ru_reg),
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
