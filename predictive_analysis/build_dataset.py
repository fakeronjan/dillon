"""
Build a per-game prediction dataset for 1999+ NFL seasons.

Joins:
  - DILLON game results (all_NFL_games.csv)
  - DILLON REACT pre-game ratings (dillon_react_ratings.csv)
  - DILLON HOTTAKE pre-game ratings (dillon_hottake_ratings.csv)
  - nflverse spreads (free CSV from github.com/nflverse/nfldata)

Output: predictive_analysis/dataset.csv, one row per game.

Conventions:
  - nflverse `spread_line` is the home spread: positive = home favored.
  - DILLON game `hca` is the home-court adjustment (2.5 normally, 0 for neutral).
  - "pre-game rating" = team's latest rating snapshot with season_week < game's season_week.
  - DILLON encodes playoffs as week 101 (WC) / 102 (DIV) / 103 (CON) / 104 (SB);
    nflverse uses 19/20/21/22. We translate nflverse -> DILLON.
"""

from pathlib import Path
import sys
import pandas as pd
import numpy as np

NFL_DIR = Path(__file__).resolve().parent.parent
SPREADS_CSV = Path("/tmp/nflverse_games.csv")
OUT_PATH = Path(__file__).resolve().parent / "dataset.csv"


# nflverse game_type -> DILLON week for playoffs
# (week numbering differs by era: pre-2021 reg season was 17 weeks, 2021+ is 18,
# so we map by game_type instead of week number.)
GAMETYPE_TO_DILLON_WEEK = {"WC": 101, "DIV": 102, "CON": 103, "SB": 104}


def code_to_name(code, season):
    """Map nflverse team code -> DILLON full team name, season-aware for relocations."""
    static = {
        "ARI": "Arizona Cardinals", "ATL": "Atlanta Falcons", "BAL": "Baltimore Ravens",
        "BUF": "Buffalo Bills", "CAR": "Carolina Panthers", "CHI": "Chicago Bears",
        "CIN": "Cincinnati Bengals", "CLE": "Cleveland Browns", "DAL": "Dallas Cowboys",
        "DEN": "Denver Broncos", "DET": "Detroit Lions", "GB": "Green Bay Packers",
        "HOU": "Houston Texans", "IND": "Indianapolis Colts", "JAX": "Jacksonville Jaguars",
        "JAC": "Jacksonville Jaguars", "KC": "Kansas City Chiefs", "MIA": "Miami Dolphins",
        "MIN": "Minnesota Vikings", "NE": "New England Patriots", "NO": "New Orleans Saints",
        "NYG": "New York Giants", "NYJ": "New York Jets", "PHI": "Philadelphia Eagles",
        "PIT": "Pittsburgh Steelers", "SEA": "Seattle Seahawks", "SF": "San Francisco 49ers",
        "TB": "Tampa Bay Buccaneers", "TEN": "Tennessee Titans",
        "WAS": "Washington Commanders", "WSH": "Washington Commanders",
    }
    if code in static:
        return static[code]
    # Relocations (DILLON keeps separate per season)
    if code in ("LA", "LAR", "STL"):
        return "Los Angeles Rams" if season >= 2016 else "St. Louis Rams"
    if code in ("LAC", "SD"):
        return "Los Angeles Chargers" if season >= 2017 else "San Diego Chargers"
    if code in ("LV", "OAK"):
        return "Las Vegas Raiders" if season >= 2020 else "Oakland Raiders"
    return None


def to_swo(season, week):
    """Season-week ordinal: ranks any (season, week) globally for asof joins."""
    return int(season) * 1000 + int(round(float(week)))


def main():
    games = pd.read_csv(NFL_DIR / "all_NFL_games.csv")
    games = games[games["season"] >= 1999].copy()
    games["swo"] = games.apply(lambda r: to_swo(r["season"], r["week"]), axis=1)

    react = pd.read_csv(NFL_DIR / "dillon_react_ratings.csv")
    react = react[react["season"] >= 1999].copy()
    react["swo"] = react.apply(lambda r: to_swo(r["season"], r["week"]), axis=1)

    hottake = pd.read_csv(NFL_DIR / "dillon_hottake_ratings.csv")
    hottake = hottake[hottake["season"] >= 1999].copy()
    hottake["swo"] = hottake.apply(lambda r: to_swo(r["season"], r["week"]), axis=1)

    spreads = pd.read_csv(SPREADS_CSV)
    spreads = spreads[spreads["season"] >= 1999].copy()
    spreads = spreads.dropna(subset=["spread_line", "home_score", "away_score"]).copy()

    # Map nflverse codes + weeks to DILLON
    spreads["home_name"] = spreads.apply(lambda r: code_to_name(r["home_team"], r["season"]), axis=1)
    spreads["away_name"] = spreads.apply(lambda r: code_to_name(r["away_team"], r["season"]), axis=1)
    spreads["dillon_week"] = spreads.apply(
        lambda r: GAMETYPE_TO_DILLON_WEEK.get(r["game_type"], int(r["week"])), axis=1)
    spreads["swo"] = spreads.apply(lambda r: to_swo(r["season"], r["dillon_week"]), axis=1)

    missing_map = spreads[spreads["home_name"].isna() | spreads["away_name"].isna()]
    if len(missing_map):
        codes = pd.unique(pd.concat([
            spreads.loc[spreads["home_name"].isna(), "home_team"],
            spreads.loc[spreads["away_name"].isna(), "away_team"],
        ]))
        print(f"WARN: {len(missing_map)} rows had unmapped codes: {sorted(codes)}", file=sys.stderr)
    spreads = spreads.dropna(subset=["home_name", "away_name"]).copy()

    # Pre-game rating lookup via merge_asof on swo (strictly less than).
    def lookup(df, ratings, side, label):
        side_col = f"{side}_name"
        ratings_sorted = ratings[["name", "swo", "rating"]].sort_values("swo")
        df_s = (df[["swo", side_col]].rename(columns={side_col: "name"})
                .reset_index().sort_values("swo"))
        merged = pd.merge_asof(
            df_s, ratings_sorted,
            on="swo", by="name",
            direction="backward", allow_exact_matches=False,
        )
        return merged.set_index("index")["rating"].rename(f"{side}_{label}")

    spreads["home_react"]   = lookup(spreads, react,   "home", "react")
    spreads["away_react"]   = lookup(spreads, react,   "away", "react")
    spreads["home_hottake"] = lookup(spreads, hottake, "home", "hottake")
    spreads["away_hottake"] = lookup(spreads, hottake, "away", "hottake")

    pre = spreads.dropna(subset=["home_react", "away_react", "home_hottake", "away_hottake"]).copy()

    # Super Bowls are at neutral sites, so DILLON and nflverse may disagree on which
    # team is "home". For SBs only, create a swapped duplicate and flip the spread sign;
    # we'll filter to whichever side matches DILLON in the merge.
    sb_swap = pre[pre["game_type"] == "SB"].copy()
    sb_swap = sb_swap.rename(columns={
        "home_name": "away_name", "away_name": "home_name",
        "home_react": "away_react", "away_react": "home_react",
        "home_hottake": "away_hottake", "away_hottake": "home_hottake",
    })
    sb_swap["spread_line"] = -sb_swap["spread_line"]
    # Recompute home/away scores after swap so the home_pts sanity check passes
    sb_swap["home_score"], sb_swap["away_score"] = sb_swap["away_score"].copy(), sb_swap["home_score"].copy()
    pre = pd.concat([pre, sb_swap], ignore_index=True)

    # Join DILLON game-level data to get HCA (handles neutral-site games like SB)
    g = games[["season", "week", "home_team_name", "visitor_team_name",
               "home_pts", "visitor_pts", "hca", "is_neutral", "unique_game_id"]].copy()
    g = g.rename(columns={
        "home_team_name": "home_name",
        "visitor_team_name": "away_name",
        "week": "dillon_week",
    })
    merged = pre.merge(
        g, on=["season", "dillon_week", "home_name", "away_name"], how="left",
    )

    # Sanity: scores agree
    score_off = merged[(merged["home_score"] != merged["home_pts"]) |
                       (merged["away_score"] != merged["visitor_pts"])]
    if len(score_off):
        print(f"WARN: {len(score_off)} score mismatches between nflverse and DILLON", file=sys.stderr)

    unmatched = merged["home_pts"].isna().sum()
    if unmatched:
        print(f"WARN: {unmatched} spread rows didn't match a DILLON game", file=sys.stderr)
    merged = merged.dropna(subset=["home_pts"]).copy()

    merged["actual_home_margin"] = merged["home_pts"] - merged["visitor_pts"]
    # nflverse spread_line: positive = home favored. So Vegas-predicted home margin = spread_line.
    merged["vegas_home_margin"] = merged["spread_line"]

    # Predicted margins
    merged["react_home_margin"]   = merged["home_react"]   - merged["away_react"]   + merged["hca"]
    merged["hottake_home_margin"] = merged["home_hottake"] - merged["away_hottake"] + merged["hca"]

    # Errors
    merged["react_err"]   = merged["react_home_margin"]   - merged["actual_home_margin"]
    merged["hottake_err"] = merged["hottake_home_margin"] - merged["actual_home_margin"]
    merged["vegas_err"]   = merged["vegas_home_margin"]   - merged["actual_home_margin"]

    # ATS bookkeeping. home_covers iff actual_margin > vegas_margin. push if equal.
    merged["home_covers"] = merged["actual_home_margin"] > merged["vegas_home_margin"]
    merged["push"]        = merged["actual_home_margin"] == merged["vegas_home_margin"]
    merged["home_wins"]   = merged["actual_home_margin"] > 0

    # Picks per rating
    for label in ("react", "hottake"):
        pred_col = f"{label}_home_margin"
        picks_col = f"{label}_picks_home"
        ats_col   = f"{label}_ats_correct"
        su_col    = f"{label}_su_correct"
        edge_col  = f"{label}_edge"
        merged[picks_col] = merged[pred_col] > merged["vegas_home_margin"]
        merged[edge_col]  = merged[pred_col] - merged["vegas_home_margin"]
        merged[ats_col]   = (
            ((merged[picks_col])  & (merged["home_covers"])) |
            ((~merged[picks_col]) & (~merged["home_covers"]))
        )
        merged.loc[merged["push"], ats_col] = pd.NA
        merged[su_col] = (merged[pred_col] > 0) == merged["home_wins"]

    # HOTTAKE-vs-REACT divergence: "fade HOTTAKE" = pick the side REACT favors when they disagree
    merged["hottake_vs_react"] = merged["hottake_home_margin"] - merged["react_home_margin"]

    keep = [
        "unique_game_id", "season", "week", "game_type", "is_neutral",
        "home_name", "away_name", "home_pts", "visitor_pts", "actual_home_margin",
        "home_react", "away_react", "home_hottake", "away_hottake", "hca",
        "react_home_margin", "hottake_home_margin", "vegas_home_margin",
        "react_err", "hottake_err", "vegas_err",
        "react_edge", "hottake_edge", "hottake_vs_react",
        "react_picks_home", "hottake_picks_home", "home_covers", "push",
        "react_ats_correct", "hottake_ats_correct",
        "react_su_correct", "hottake_su_correct", "home_wins",
    ]
    out = merged[keep].sort_values(["season", "week", "unique_game_id"]).reset_index(drop=True)
    out.to_csv(OUT_PATH, index=False)
    print(f"Wrote {len(out)} games -> {OUT_PATH}")
    print(f"  Seasons: {sorted(out['season'].unique())}")
    print(f"  Game types: {out['game_type'].value_counts().to_dict()}")
    print(f"  Pushes: {int(out['push'].sum())}")


if __name__ == "__main__":
    main()
