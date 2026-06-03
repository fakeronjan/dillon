# =========================================================
# DILLON NFL POWER RATINGS
# =========================================================

import requests
import pandas as pd
import numpy as np
from datetime import datetime

# =========================================================
# CONFIGURATION
# =========================================================

MIN_SEASON           = 1970   # first post-AFL/NFL-merger season
WEEKS_REACT          = 20     # rolling window for REACT ratings (long-view)
HOME_FIELD_ADVANTAGE = 2.5    # raw-point home advantage; subtracted from home margin pre-transform
MARGIN_CAP           = 35     # symmetric clip on the HCA-adjusted home margin

# Margin transform: cap clips margins symmetrically at MARGIN_CAP. Ratings
# read as approximate point-spread vs avg team.
MARGIN_TRANSFORM = "cap"

# WLS: weights affect observation influence, not margin magnitude.
WEIGHTING_MODE = "wls"

# Special week numbers used for playoff rounds in our data
WEEK_WILDCARD   = 101
WEEK_DIVISION   = 102
WEEK_CONFCHAMP  = 103
WEEK_SUPERBOWL  = 104

# Same-market rebrand consolidation. Maps historical team names to current
# canonical names so a single franchise's history reads as one team across
# rebrands. RELOCATIONS are deliberately kept separate ("move the team =
# lose the history" policy): Houston Oilers, Baltimore Colts, San Diego
# Chargers, LA Raiders, Oakland Raiders, St. Louis Cardinals, St. Louis
# Rams etc. all remain distinct from their post-move franchises.
TEAM_ALIASES = {
    'Boston Patriots':           'New England Patriots',
    'Washington Redskins':       'Washington Commanders',
    'Washington Football Team':  'Washington Commanders',
    'Phoenix Cardinals':         'Arizona Cardinals',
    'Tennessee Oilers':          'Tennessee Titans',
}


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

    # Apply same-market rebrand consolidation before any team-keyed work
    # (margins, result strings, name_season, downstream merges). Old names
    # show up rendered as the franchise's current name everywhere.
    df['winner'] = df['winner'].replace(TEAM_ALIASES)
    df['loser']  = df['loser'].replace(TEAM_ALIASES)

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

    # Decode 'home' field into venue-aware home/visitor team mapping for the
    # solver. Original convention: NaN = winner played at home; '@' = winner
    # was the visitor (loser at home); 'N' = neutral site.
    df['home'] = df['home'].fillna('H')
    df['home'] = df['home'].replace({'@': 'V', 'N': 'N'})  # H = winner home, V = winner away, N = neutral

    # Home/visitor teams + points (used by the WLS solver).
    df['home_team_name']    = np.where(df['home'] == 'H', df['winner'], df['loser'])
    df['visitor_team_name'] = np.where(df['home'] == 'H', df['loser'],  df['winner'])
    df['home_pts']          = np.where(df['home'] == 'H', df['ptsw'],   df['ptsl'])
    df['visitor_pts']       = np.where(df['home'] == 'H', df['ptsl'],   df['ptsw'])

    # Per-game HCA: 0 for neutral sites, HOME_FIELD_ADVANTAGE otherwise.
    df['is_neutral'] = (df['home'] == 'N').astype(int)
    df['hca']        = np.where(df['is_neutral'] == 1, 0.0, HOME_FIELD_ADVANTAGE)

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
# FAKERONJAN WLS RATINGS — homebrew weighted least squares solver
# =========================================================

def _apply_margin_transform(margin, transform, cap):
    """Sign-preserving transform applied to (raw_margin - hca)."""
    m = np.asarray(margin, dtype=float)
    if transform == "raw":
        return m
    if transform == "sqrt":
        return np.sign(m) * np.sqrt(np.abs(m))
    if transform == "cap":
        return np.clip(m, -cap, cap)
    if transform == "log":
        return np.sign(m) * np.log1p(np.abs(m))
    if transform == "tanh":
        return cap * np.tanh(m / cap)
    raise ValueError(f"Unknown MARGIN_TRANSFORM: {transform}")


def _solve_wls(window_df, weighting_mode, margin_transform, margin_cap):
    """
    Solve for team fakeronjan WLS ratings on a single rolling window.

    Builds X (n_games × n_teams) with +1 for home, -1 for visitor, y from
    the transformed HCA-adjusted home margin, and W from the recency
    weights. Solves min sum_i w_i * (X_i r - y_i)^2 with a zero-sum
    constraint enforced as an extra high-weight row.

    HCA is per-game (column 'hca' in window_df) — supports neutral-site
    games (Super Bowl, international games) where HCA should be 0.
    """
    teams = sorted(set(window_df["home_team_name"]) | set(window_df["visitor_team_name"]))
    team_idx = {t: i for i, t in enumerate(teams)}
    n_teams = len(teams)
    n_games = len(window_df)

    X = np.zeros((n_games + 1, n_teams))
    y = np.zeros(n_games + 1)
    w = np.zeros(n_games + 1)

    home_pts    = window_df["home_pts"].to_numpy(dtype=float)
    visitor_pts = window_df["visitor_pts"].to_numpy(dtype=float)
    hca         = window_df["hca"].to_numpy(dtype=float)
    weights     = window_df["date_weight"].to_numpy(dtype=float)
    home_names    = window_df["home_team_name"].to_numpy()
    visitor_names = window_df["visitor_team_name"].to_numpy()

    raw_margin  = home_pts - visitor_pts - hca
    transformed = _apply_margin_transform(raw_margin, margin_transform, margin_cap)

    for i in range(n_games):
        X[i, team_idx[home_names[i]]] = 1.0
        X[i, team_idx[visitor_names[i]]] = -1.0

    if weighting_mode == "wls":
        y[:n_games] = transformed
        w[:n_games] = weights
    elif weighting_mode == "margin_scale":
        y[:n_games] = transformed * weights
        w[:n_games] = 1.0
    else:
        raise ValueError(f"Unknown WEIGHTING_MODE: {weighting_mode}")

    # Zero-sum constraint via high-weight extra row.
    X[-1, :] = 1.0
    y[-1] = 0.0
    w[-1] = 1.0e8

    sqrt_w = np.sqrt(w)
    Xw = X * sqrt_w[:, None]
    yw = y * sqrt_w
    r, *_ = np.linalg.lstsq(Xw, yw, rcond=None)

    out = pd.DataFrame({"name": teams, "rating": r})
    out["rank"] = out["rating"].rank(ascending=False, method="min").astype(int)
    return out


def _solve_wls_od(window_df, weighting_mode, hca_off_share=0.5):
    """
    Solve for team OFFENSIVE + DEFENSIVE WLS ratings (Massey-style split).

    Each game contributes 2 rows:
      home_O - visitor_D = home_pts - mu - hca * off_share
      visitor_O - home_D = visitor_pts - mu + hca * def_share

    Where mu = league mean points / team / game across the window, so O and D
    are centered on zero. hca_off_share allocates the home-field advantage
    between offense (h_off) and defense (h_def). 0.5 = 50/50 split.

    Parameter vector layout: [O_1..O_n, D_1..D_n]. Two zero-sum constraint
    rows pin both vectors. A team's net contribution to point margin is
    REACT_team = O_team + D_team; the home-vs-visitor margin then satisfies
    margin = (home_REACT - visitor_REACT) + HCA, matching the single-rating
    solver's interpretation.

    No half-equation cap on first pass (the single-rating solver's MARGIN_CAP
    operates on net margin, which has no clean per-side analog). If garbage-
    time blowouts contaminate rankings we can revisit.
    """
    teams = sorted(set(window_df["home_team_name"]) | set(window_df["visitor_team_name"]))
    team_idx = {t: i for i, t in enumerate(teams)}
    n_teams = len(teams)
    n_games = len(window_df)

    home_pts    = window_df["home_pts"].to_numpy(dtype=float)
    visitor_pts = window_df["visitor_pts"].to_numpy(dtype=float)
    hca         = window_df["hca"].to_numpy(dtype=float)
    weights     = window_df["date_weight"].to_numpy(dtype=float)
    home_names    = window_df["home_team_name"].to_numpy()
    visitor_names = window_df["visitor_team_name"].to_numpy()

    mu = (home_pts.sum() + visitor_pts.sum()) / (2 * n_games)
    h_off_share = float(hca_off_share)
    h_def_share = 1.0 - h_off_share

    # 2 rows per game + 2 zero-sum constraint rows
    X = np.zeros((2 * n_games + 2, 2 * n_teams))
    y = np.zeros(2 * n_games + 2)
    w = np.zeros(2 * n_games + 2)

    for i in range(n_games):
        h_idx = team_idx[home_names[i]]
        v_idx = team_idx[visitor_names[i]]
        # Half-equation: home_O - visitor_D = home_pts - mu - hca*h_off_share
        X[2*i,     h_idx]              = 1.0          # +home_O
        X[2*i,     n_teams + v_idx]    = -1.0         # -visitor_D
        y_home = home_pts[i] - mu - hca[i] * h_off_share
        # Half-equation: visitor_O - home_D = visitor_pts - mu + hca*h_def_share
        X[2*i + 1, v_idx]              = 1.0          # +visitor_O
        X[2*i + 1, n_teams + h_idx]    = -1.0         # -home_D
        y_vis  = visitor_pts[i] - mu + hca[i] * h_def_share

        if weighting_mode == "wls":
            y[2*i]     = y_home
            y[2*i + 1] = y_vis
            w[2*i]     = weights[i]
            w[2*i + 1] = weights[i]
        elif weighting_mode == "margin_scale":
            y[2*i]     = y_home * weights[i]
            y[2*i + 1] = y_vis  * weights[i]
            w[2*i]     = 1.0
            w[2*i + 1] = 1.0
        else:
            raise ValueError(f"Unknown WEIGHTING_MODE: {weighting_mode}")

    # Zero-sum on O
    X[-2, :n_teams] = 1.0
    y[-2] = 0.0
    w[-2] = 1.0e8
    # Zero-sum on D
    X[-1, n_teams:] = 1.0
    y[-1] = 0.0
    w[-1] = 1.0e8

    sqrt_w = np.sqrt(w)
    Xw = X * sqrt_w[:, None]
    yw = y * sqrt_w
    sol, *_ = np.linalg.lstsq(Xw, yw, rcond=None)

    out = pd.DataFrame({
        "name":     teams,
        "rating_o": sol[:n_teams],
        "rating_d": sol[n_teams:],
    })
    out["rank_o"] = out["rating_o"].rank(ascending=False, method="min").astype(int)
    out["rank_d"] = out["rating_d"].rank(ascending=False, method="min").astype(int)
    return out


def compute_ratings(master_df, existing_ratings_df, window, label, compute_od=False):
    """
    Compute fakeronjan WLS ratings using a rolling `window`-week window.
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

        win['date_weight'] = (win['cume_week_id'] - i + window) / window

        current_week = win['season_week'].max()
        season       = int(win['season'].max())

        ranked = _solve_wls(
            win,
            weighting_mode=WEIGHTING_MODE,
            margin_transform=MARGIN_TRANSFORM,
            margin_cap=MARGIN_CAP,
        )
        if compute_od:
            ranked_od = _solve_wls_od(win, weighting_mode=WEIGHTING_MODE)
            ranked = ranked.merge(ranked_od, on='name', how='left')
            # REACT-calibrated O/D: shift each team's (O_raw, D_raw) by the same
            # delta so O+D == REACT exactly. The shape of the split (O - D, i.e.
            # the offensive vs defensive lean) is preserved; only the absolute
            # level is anchored to REACT. This means REACT's MARGIN_CAP — which
            # handles garbage-time blowouts in the single-rating fit — carries
            # through to O/D too, without needing a separate half-equation cap.
            #
            # delta = (REACT - O_raw - D_raw) / 2  per team
            # O = O_raw + delta;  D = D_raw + delta
            # => O + D = O_raw + D_raw + 2*delta = REACT
            # => O - D = O_raw - D_raw (split preserved)
            # Zero-sum stays intact because sum(delta) = 0 (all three vectors
            # are zero-sum across the league).
            delta = (ranked['rating'] - ranked['rating_o'] - ranked['rating_d']) / 2.0
            ranked['rating_o'] = ranked['rating_o'] + delta
            ranked['rating_d'] = ranked['rating_d'] + delta
            ranked['rank_o'] = ranked['rating_o'].rank(ascending=False, method='min').astype(int)
            ranked['rank_d'] = ranked['rating_d'].rank(ascending=False, method='min').astype(int)
        ranked['season_week'] = current_week
        ranked['ranking_id']  = i
        ranked['season']      = season
        new_frames.append(ranked)

    df = pd.concat([existing_ratings_df] + new_frames, axis=0, sort=False).reset_index(drop=True)
    # Empty-cache reads land columns as object dtype; coerce to numeric so
    # downstream merges + assemble_final's round/astype(int) work cleanly.
    for col in ('season', 'ranking_id', 'season_week'):
        df[col] = pd.to_numeric(df[col])
    df['season'] = df['season'].astype(int)
    df['ranking_id'] = df['ranking_id'].astype(int)
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
    min_date_id = WEEKS_REACT

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

def assemble_final(master_df, react_df, standings_df):
    """Merge REACT ratings + standings, add flags."""
    print('Final step — merging DILLON ratings and standings...')

    final_df = pd.merge(react_df, standings_df, how='left', on=['ranking_id', 'name'])
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

    final_df['lastgame']  = (final_df['winner_last_game'] + final_df['loser_last_game'])
    final_df['opponent']  = final_df['loser'] + final_df['winner']

    # Drop teams with no games yet (handles expansion / pre-merger lineage edge cases)
    final_df = final_df[final_df['record'] != '0-0']

    final_df = final_df[[
        'ranking_id', 'season_week', 'season', 'week', 'name', 'name_season',
        'rating', 'rank',
        'rating_o', 'rank_o', 'rating_d', 'rank_d',
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
    react_df = compute_ratings(master_df, existing_react, WEEKS_REACT, 'REACT', compute_od=True)
    react_df.to_csv('dillon_react_ratings.csv', index=False)

    # 4. Standings
    try:
        existing_standings = pd.read_csv('weekly_standings.csv')
    except FileNotFoundError:
        existing_standings = pd.DataFrame()
    standings_df = compute_standings(master_df, existing_standings)
    standings_df.to_csv('weekly_standings.csv', index=False)

    # 5. Final assembly
    assemble_final(master_df, react_df, standings_df)
