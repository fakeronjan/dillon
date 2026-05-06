# =========================================================
# DILLON NFL POWER RATINGS
# =========================================================

import requests
import pandas as pd
import numpy as np
# rankit==0.2 uses deprecated numpy aliases (np.int, np.float, np.bool) removed in numpy 1.24+.
# Restore them before rankit import so the Massey solver works.
if not hasattr(np, 'int'):   np.int = int
if not hasattr(np, 'float'): np.float = float
if not hasattr(np, 'bool'):  np.bool = bool
from datetime import datetime
import warnings
import rankit  # pip install rankit
from rankit.Table import Table
from rankit.Ranker import MasseyRanker

# Suppress SettingWithCopyWarning from rankit library internals.
# pandas 3.x removed this class, so guard the lookup.
try:
    warnings.filterwarnings('ignore', category=pd.errors.SettingWithCopyWarning)
except AttributeError:
    pass

# =========================================================
# CONFIGURATION
# =========================================================

MIN_SEASON           = 1970   # first post-AFL/NFL-merger season
WEEKS_REACT          = 20     # rolling window for REACT ratings (long-view)
WEEKS_HOTTAKE        = 10     # rolling window for HOTTAKE ratings (recent-form)
HOME_FIELD_ADVANTAGE = 0.5
MARGIN_CAP           = 35

# Special week numbers used for playoff rounds in our data
WEEK_WILDCARD   = 101
WEEK_DIVISION   = 102
WEEK_CONFCHAMP  = 103
WEEK_SUPERBOWL  = 104


# =========================================================
# SCRAPING
# =========================================================

def scrape_games(min_season, max_season, existing_df):
    """
    Scrape any seasons not already fully captured in existing_df.
    Returns combined DataFrame, saved to loaded_NFL_games.csv.
    """
    max_season_completed = max(existing_df['Season']) - 1   # latest season may be partial
    min_season_completed = min(existing_df['Season'])

    print(f"Already have complete data for seasons {min_season_completed}-{max_season_completed}")
    print(f"Checking for new data through season {max_season}")

    new_frames = []
    for year in range(max_season_completed + 1, max_season + 1):
        url = f'https://www.pro-football-reference.com/years/{year}/games.htm'
        try:
            df = pd.read_html(url)[0]
        except Exception:
            print(f"{year} — not found, skipping.")
            continue
        df['Season'] = year
        # pro-football-reference uses 'Pts' and 'Pts.1' (winner / loser points)
        df.rename(columns={'Pts': 'PtsW', 'Pts.1': 'PtsL'}, inplace=True)
        new_frames.append(df)
        print(f"{year} — scraped!")

    print("Successfully scraped!")

    combined = pd.concat([existing_df] + new_frames, axis=0, sort=False).reset_index(drop=True)
    combined.sort_values(['Season', 'Week', 'Winner/tie', 'Loser/tie'], inplace=True)
    combined.drop_duplicates(subset=['Season', 'Week', 'Winner/tie', 'Loser/tie'], keep='last', inplace=True)
    combined.to_csv('loaded_NFL_games.csv', index=False)
    return combined


# =========================================================
# GAME DATA PREPARATION
# =========================================================

def prepare_game_data(raw_df):
    """
    Clean and enrich the raw games DataFrame with margins, win flags,
    adjusted scores, week IDs, and result strings.
    """
    df = raw_df.copy()

    # Standardize columns. pro-football-reference's table:
    # ['Week','Day','Date','Time','Winner/tie','at','Loser/tie','boxscore','PtsW','PtsL','YdsW','TOW','YdsL','TOL','Season']
    df.columns = ['week', 'day', 'date', 'time', 'winner', 'home', 'loser',
                  'boxscore', 'ptsw', 'ptsl', 'ydsw', 'tow', 'ydsl', 'tol', 'season']

    df = df[['season', 'week', 'date', 'winner', 'loser', 'ptsw', 'ptsl', 'home']].copy()
    df = df[df['date'] != 'Date']
    df = df[df['date'] != 'Playoffs']
    df = df.dropna(subset=['ptsw'])

    df['ptsw'] = pd.to_numeric(df['ptsw'])
    df['ptsl'] = pd.to_numeric(df['ptsl'])

    # Margin and win flags (ties register 0.5 for both sides)
    df['marginw']  = df['ptsw'] - df['ptsl']
    df['marginl']  = -df['marginw']
    df['is_tie']   = (df['ptsw'] == df['ptsl']).astype(int)
    df['winw']     = np.where(df['ptsw'] > df['ptsl'], 1, 0.5)
    df['winl']     = 1 - df['winw']

    # Recode special weeks
    df['week'] = df['week'].replace({
        'WildCard':  WEEK_WILDCARD,
        'Division':  WEEK_DIVISION,
        'ConfChamp': WEEK_CONFCHAMP,
        'SuperBowl': WEEK_SUPERBOWL,
    })
    df['week'] = pd.to_numeric(df['week'])

    # Last-game result strings (T = tie, W = win, L = loss)
    df['winner_home'] = np.where(df['home'] == '@', ' @ ', ' vs. ')
    df['winner_home'] = np.where(df['home'] == 'N', ' vs. (N) ', df['winner_home'])
    df['winner_last_game'] = np.where(
        df['winw'] == 0.5,
        'T ' + df['ptsw'].map(str) + '-' + df['ptsl'].map(str) + df['winner_home'] + df['loser'],
        'W ' + df['ptsw'].map(str) + '-' + df['ptsl'].map(str) + df['winner_home'] + df['loser']
    )
    df['loser_home'] = np.where(df['home'] == '@', ' vs. ', ' @ ')
    df['loser_home'] = np.where(df['home'] == 'N', ' vs. (N) ', df['loser_home'])
    df['loser_last_game'] = np.where(
        df['winl'] == 0.5,
        'T ' + df['ptsl'].map(str) + '-' + df['ptsw'].map(str) + df['loser_home'] + df['winner'],
        'L ' + df['ptsl'].map(str) + '-' + df['ptsw'].map(str) + df['loser_home'] + df['winner']
    )

    # Home-field-adjusted margin
    df['home'] = df['home'].fillna(-HOME_FIELD_ADVANTAGE)
    df['home'] = df['home'].replace({'@': HOME_FIELD_ADVANTAGE, 'N': 0})
    df['home'] = pd.to_numeric(df['home'])
    df['adjmarginw'] = (df['marginw'] + df['home']).clip(upper=MARGIN_CAP)
    df['adjmarginl'] = -df['adjmarginw']

    # IDs
    df['week_id']        = df.groupby(['week']).ngroup() + 1
    df['season_week']    = df['season'] + df['week'] / 1000
    df['cume_week_id']   = df.groupby(['season_week']).ngroup() + 1
    df['unique_game_id'] = df.groupby(df.columns.tolist(), sort=False).ngroup() + 1

    # team-season identifiers (used downstream for SB matching)
    df['winner_season'] = df['winner'] + ' - ' + df['season'].map(str)
    df['loser_season']  = df['loser']  + ' - ' + df['season'].map(str)

    df = df.drop_duplicates(keep='first').copy()
    df.to_csv('all_NFL_games.csv', index=False)
    print('CSV of NFL games is ready!')
    return df


# =========================================================
# MASSEY RATINGS
# =========================================================

def compute_ratings(master_df, existing_ratings_df, window, label):
    """
    Compute Massey ratings using a rolling `window`-week window.
    Skips ranking_ids already present in existing_ratings_df.
    """
    max_date_id = int(master_df['cume_week_id'].max())
    min_date_id = window

    if len(existing_ratings_df) > 0 and 'ranking_id' in existing_ratings_df.columns:
        max_ranked = int(existing_ratings_df['ranking_id'].max())
        min_ranked = int(existing_ratings_df['ranking_id'].min())
    else:
        max_ranked = -1
        min_ranked = -1

    print(f'Running {label} ratings ({window}-week window)...')
    new_frames = []

    for i in range(min_date_id, max_date_id + 1):
        # Always recompute the latest cached week_id so mid-week games (MNF/TNF
        # arriving after Sunday's run) get folded into that week's rating.
        if min_ranked <= i < max_ranked:
            continue

        win = master_df[
            (master_df['cume_week_id'] >= i - (window - 1)) &
            (master_df['cume_week_id'] <= i)
        ].copy()

        win['date_weight']     = (win['cume_week_id'] - i + window) / window
        win['weightedmarginl'] = win['adjmarginl'] * win['date_weight']
        win['weightedmarginw'] = -win['weightedmarginl']

        current_week = win['season_week'].max()
        season       = int(win['season'].max())

        nfl_table = Table(win, ['loser', 'winner', 'weightedmarginl', 'weightedmarginw'])
        ranked = MasseyRanker(nfl_table).rank()
        ranked['season_week'] = current_week
        ranked['ranking_id']  = i
        ranked['season']      = season
        new_frames.append(ranked)

    df = pd.concat([existing_ratings_df] + new_frames, axis=0, sort=False).reset_index(drop=True)
    df['week'] = (df['season_week'] - df['season']) * 1000
    df.sort_values(['ranking_id', 'name'], inplace=True)
    # Dedupe by (ranking_id, name) keeping the freshly computed row.
    # The latest ranking_id is always recomputed (mid-week games like MNF/TNF),
    # so we may have both an existing row and a new row for it. Without a
    # subset, drop_duplicates misses these because float precision differs.
    df.drop_duplicates(subset=['ranking_id', 'name'], keep='last', inplace=True)
    print(f'CSV of {label} ratings is ready!')
    return df


# =========================================================
# STANDINGS (W-L-T tracking, ties counted separately)
# =========================================================

def _make_pivot(df, value_col, index_col, new_value_name, aggfunc=np.sum):
    pivot = pd.pivot_table(df, values=value_col, index=[index_col], aggfunc=aggfunc)
    return pivot.fillna(0).reset_index().rename(columns={value_col: new_value_name, index_col: 'name'})


def compute_standings(master_df, existing_standings_df):
    """
    Compute cumulative season W-L-T standings per week.
    Skips ranking_ids already present in existing_standings_df.
    """
    df_for_calc = master_df[['season', 'season_week', 'cume_week_id', 'winner', 'loser', 'is_tie']].copy()
    # Add explicit per-side tracking so ties count separately from wins/losses
    df_for_calc['winner_real_win']  = np.where(df_for_calc['is_tie'] == 0, 1, 0)
    df_for_calc['loser_real_loss']  = np.where(df_for_calc['is_tie'] == 0, 1, 0)
    df_for_calc['winner_tie']       = df_for_calc['is_tie']
    df_for_calc['loser_tie']        = df_for_calc['is_tie']

    max_date_id = int(master_df['cume_week_id'].max())
    min_date_id = min(WEEKS_REACT, WEEKS_HOTTAKE)

    if len(existing_standings_df) > 0 and 'ranking_id' in existing_standings_df.columns:
        max_ranked = int(existing_standings_df['ranking_id'].max())
        min_ranked = int(existing_standings_df['ranking_id'].min())
    else:
        max_ranked = -1
        min_ranked = -1

    print('Producing standings...')
    new_frames = []

    for i in range(min_date_id, max_date_id + 1):
        # Always recompute the latest cached week_id so mid-week games (MNF/TNF
        # arriving after Sunday's run) get folded into that week's rating.
        if min_ranked <= i < max_ranked:
            continue

        slicer = df_for_calc[df_for_calc['cume_week_id'] <= i]
        season = int(slicer['season'].max())
        slicer = slicer[slicer['season'] == season]
        ranking_week = slicer['season_week'].max()

        wins_w  = _make_pivot(slicer, 'winner_real_win', 'winner', 'wins_as_winner')
        loss_l  = _make_pivot(slicer, 'loser_real_loss', 'loser',  'losses_as_loser')
        tie_w   = _make_pivot(slicer, 'winner_tie',      'winner', 'ties_as_winner')
        tie_l   = _make_pivot(slicer, 'loser_tie',       'loser',  'ties_as_loser')

        merged = (wins_w.merge(loss_l, on='name', how='outer')
                        .merge(tie_w,  on='name', how='outer')
                        .merge(tie_l,  on='name', how='outer')
                        .fillna(0))

        merged['wins']   = merged['wins_as_winner'].astype(int)
        merged['losses'] = merged['losses_as_loser'].astype(int)
        merged['ties']   = (merged['ties_as_winner'] + merged['ties_as_loser']).astype(int)

        # Format: "10-5-1" if ties exist for the team, otherwise "10-5"
        def _fmt(row):
            if row['ties'] > 0:
                return f"{row['wins']}-{row['losses']}-{row['ties']}"
            return f"{row['wins']}-{row['losses']}"
        merged['record'] = merged.apply(_fmt, axis=1)

        merged = merged[['name', 'wins', 'losses', 'ties', 'record']]
        merged['ranking_id']   = i
        merged['season_week']  = ranking_week
        merged['season']       = season
        new_frames.append(merged)

    df = pd.concat([existing_standings_df] + new_frames, axis=0, sort=False).reset_index(drop=True)
    df.sort_values(['ranking_id', 'name'], inplace=True)
    # Same dedupe approach as compute_ratings — the latest ranking_id is always
    # recomputed, so the existing row and new row coexist after concat.
    df.drop_duplicates(subset=['ranking_id', 'name'], keep='last', inplace=True)
    print('CSV of standings is ready!')
    return df


# =========================================================
# FINAL ASSEMBLY
# =========================================================

def assemble_final(master_df, react_df, hottake_df, standings_df):
    """Merge REACT + HOTTAKE ratings + standings, add flags."""
    print('Final step — merging DILLON ratings and standings...')

    # Merge the two rating streams
    both = pd.merge(react_df, hottake_df, how='left',
                    on=['ranking_id', 'season_week', 'season', 'name', 'week'])
    both.rename(columns={
        'rating_x': 'rating',  'rating_y': 'rating2',
        'rank_x':   'rank',    'rank_y':   'rank2',
    }, inplace=True)

    final_df = pd.merge(both, standings_df, how='left', on=['ranking_id', 'name'])
    final_df.rename(columns={'season_week_x': 'season_week', 'season_x': 'season'}, inplace=True)
    final_df['season'] = final_df['season'].round(0).astype(int)
    final_df['record'] = final_df['record'].fillna('0-0')

    final_df['week']        = ((final_df['season_week'] - final_df['season']) * 1000).round(0).astype(int)
    final_df['name_season'] = final_df['name'] + ' - ' + final_df['season'].map(str)

    latest_week_id = final_df['ranking_id'].max()
    final_df['most_recent_week'] = (final_df['ranking_id'] == latest_week_id).astype(int)

    # season_flag: only populated for fully-complete seasons.
    # NFL season YYYY ends in Feb of YYYY+1; "fully complete" once today is past March 31 of YYYY+1.
    today = datetime.now().date()
    def season_is_fully_complete(season):
        return today > datetime(int(season) + 1, 3, 31).date()

    seasons = sorted(final_df['season'].unique())

    # Always compute "last regular-season week" so we know where playoffs begin,
    # even for in-progress seasons (used for filtering on the frontend).
    final_df['last_week_of_regular_season'] = 0
    for s in seasons:
        season_rows = final_df[final_df['season'] == s]
        reg = season_rows[season_rows['week'] < 100]
        if reg.empty:
            continue
        max_reg_week = reg['season_week'].max()
        final_df.loc[final_df['season_week'] == max_reg_week, 'last_week_of_regular_season'] = 1

    # season_flag: 0 = regular, 1 = last regular-season week, 2 = Super Bowl week
    final_df['season_flag'] = 0
    for s in seasons:
        if not season_is_fully_complete(s):
            continue
        season_rows = final_df[final_df['season'] == s]
        reg = season_rows[season_rows['week'] < 100]
        if not reg.empty:
            max_reg_week = reg['season_week'].max()
            final_df.loc[
                (final_df['season'] == s) & (final_df['season_week'] == max_reg_week),
                'season_flag'
            ] = 1
        final_df.loc[
            (final_df['season'] == s) & (final_df['week'] == WEEK_SUPERBOWL),
            'season_flag'
        ] = 2

    # Super Bowl champ + runner-up
    final_df['sb_champ']    = 0
    final_df['sb_runnerup'] = 0
    for s in seasons:
        sg = master_df[master_df['season'] == s]
        if sg.empty:
            continue
        if sg['week'].max() != WEEK_SUPERBOWL:
            continue
        sb_game = sg[sg['week'] == WEEK_SUPERBOWL]
        if sb_game.empty:
            continue
        winner_season = sb_game['winner_season'].iloc[0]
        loser_season  = sb_game['loser_season'].iloc[0]
        final_df.loc[final_df['name_season'] == winner_season, 'sb_champ']    = 1
        final_df.loc[final_df['name_season'] == loser_season,  'sb_runnerup'] = 1

    final_df['sb_status'] = final_df['sb_runnerup'] + 2 * final_df['sb_champ']

    # Last game info per (season, week, team)
    lastgamew = master_df[['season', 'week', 'winner', 'winner_last_game', 'loser']].rename(columns={'winner': 'name'})
    lastgamel = master_df[['season', 'week', 'loser',  'loser_last_game',  'winner']].rename(columns={'loser':  'name'})
    final_df = final_df.merge(lastgamew, how='left', on=['season', 'week', 'name'])
    final_df = final_df.merge(lastgamel, how='left', on=['season', 'week', 'name'])

    for col in ['winner_last_game', 'loser_last_game', 'winner', 'loser']:
        final_df[col] = final_df[col].fillna('')

    final_df['lastgame']  = (final_df['winner_last_game'] + final_df['loser_last_game']).replace('', 'Bye / No Game')
    final_df['opponent']  = final_df['loser'] + final_df['winner']

    # Drop teams with no games yet (handles expansion / pre-merger lineage edge cases)
    final_df = final_df[final_df['record'] != '0-0']

    final_df = final_df[[
        'ranking_id', 'season_week', 'season', 'week', 'name', 'name_season',
        'rating', 'rank', 'rating2', 'rank2',
        'record', 'most_recent_week', 'last_week_of_regular_season',
        'season_flag', 'sb_champ', 'sb_runnerup', 'sb_status',
        'lastgame', 'opponent'
    ]]

    final_df.to_csv('dillon_ratings_with_standings.csv', index=False)
    print('CSV of everything is ready!')
    return final_df


# =========================================================
# MAIN
# =========================================================

if __name__ == '__main__':
    max_season = datetime.now().year + 1

    # 1. Scrape
    existing_games = pd.read_csv('loaded_NFL_games.csv')
    raw_df = scrape_games(MIN_SEASON, max_season, existing_games)

    # 2. Prepare game data
    master_df = prepare_game_data(raw_df)

    # 3. REACT ratings
    try:
        existing_react = pd.read_csv('dillon_react_ratings.csv')
    except FileNotFoundError:
        existing_react = pd.DataFrame()
    react_df = compute_ratings(master_df, existing_react, WEEKS_REACT, 'REACT')
    react_df.to_csv('dillon_react_ratings.csv', index=False)

    # 4. HOTTAKE ratings
    try:
        existing_hottake = pd.read_csv('dillon_hottake_ratings.csv')
    except FileNotFoundError:
        existing_hottake = pd.DataFrame()
    hottake_df = compute_ratings(master_df, existing_hottake, WEEKS_HOTTAKE, 'HOTTAKE')
    hottake_df.to_csv('dillon_hottake_ratings.csv', index=False)

    # 5. Standings
    try:
        existing_standings = pd.read_csv('weekly_standings.csv')
    except FileNotFoundError:
        existing_standings = pd.DataFrame()
    standings_df = compute_standings(master_df, existing_standings)
    standings_df.to_csv('weekly_standings.csv', index=False)

    # 6. Final assembly
    assemble_final(master_df, react_df, hottake_df, standings_df)
