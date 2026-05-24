"""
Analyze DILLON predictive accuracy: REACT + HOTTAKE vs Vegas spread.

Headline questions:
  1. How does each rating's predictive accuracy compare to Vegas?
  2. Which rating (REACT or HOTTAKE) carries more signal?
  3. Is HOTTAKE a CONTRARIAN signal? When HOTTAKE diverges from REACT, does
     fading HOTTAKE / backing REACT produce ATS edge?
  4. Does any of this change in the playoffs?
"""

from pathlib import Path
import numpy as np
import pandas as pd

DATA = Path(__file__).resolve().parent / "dataset.csv"


def fmt_pct(x):
    return f"{x*100:.1f}%" if pd.notna(x) else "n/a"


def section(title):
    print()
    print("=" * 72)
    print(title)
    print("=" * 72)


def summary(df, label):
    if len(df) == 0:
        print(f"{label}: empty"); return
    rmse_v = np.sqrt(np.mean(df["vegas_err"] ** 2))
    mae_v  = np.mean(np.abs(df["vegas_err"]))
    rmse_r = np.sqrt(np.mean(df["react_err"] ** 2))
    mae_r  = np.mean(np.abs(df["react_err"]))
    rmse_h = np.sqrt(np.mean(df["hottake_err"] ** 2))
    mae_h  = np.mean(np.abs(df["hottake_err"]))
    su_r = df["react_su_correct"].mean()
    su_h = df["hottake_su_correct"].mean()
    home_win_rate = df["home_wins"].mean()
    ats_df = df[~df["push"]]
    ats_r = ats_df["react_ats_correct"].mean()
    ats_h = ats_df["hottake_ats_correct"].mean()
    print(f"{label}  n={len(df)}")
    print(f"  Margin RMSE:  Vegas {rmse_v:.2f}  |  REACT {rmse_r:.2f}  |  HOTTAKE {rmse_h:.2f}")
    print(f"  Margin MAE:   Vegas {mae_v:.2f}   |  REACT {mae_r:.2f}   |  HOTTAKE {mae_h:.2f}")
    print(f"  SU accuracy:                |  REACT {fmt_pct(su_r)}   |  HOTTAKE {fmt_pct(su_h)}   (home base {fmt_pct(home_win_rate)})")
    print(f"  ATS vs Vegas:               |  REACT {fmt_pct(ats_r)}   |  HOTTAKE {fmt_pct(ats_h)}   ({len(ats_df)} bets)")


def by_season(df):
    section("Per-season ATS (excludes pushes)")
    rows = []
    for season, g in df.groupby("season"):
        gns = g[~g["push"]]
        n = len(gns)
        if n == 0: continue
        rows.append({
            "season": int(season),
            "n": n,
            "REACT_RMSE":  np.sqrt(np.mean(g["react_err"] ** 2)),
            "HOTTAKE_RMSE": np.sqrt(np.mean(g["hottake_err"] ** 2)),
            "Vegas_RMSE":  np.sqrt(np.mean(g["vegas_err"] ** 2)),
            "REACT_ATS":   gns["react_ats_correct"].mean(),
            "HOTTAKE_ATS": gns["hottake_ats_correct"].mean(),
        })
    out = pd.DataFrame(rows)
    print(out.to_string(index=False, formatters={
        "REACT_RMSE":  "{:.2f}".format,
        "HOTTAKE_RMSE":"{:.2f}".format,
        "Vegas_RMSE":  "{:.2f}".format,
        "REACT_ATS":   "{:.1%}".format,
        "HOTTAKE_ATS": "{:.1%}".format,
    }))


def calibration(df, label, pred_col):
    section(f"{label} calibration: predicted home margin -> actual mean home margin")
    bins = [-np.inf, -14, -10, -7, -3, 0, 3, 7, 10, 14, np.inf]
    df = df.copy()
    df["bucket"] = pd.cut(df[pred_col], bins=bins)
    cal = df.groupby("bucket", observed=True).agg(
        n=(pred_col, "size"),
        predicted=(pred_col, "mean"),
        actual=("actual_home_margin", "mean"),
        home_win=("home_wins", "mean"),
    )
    cal["resid"] = cal["actual"] - cal["predicted"]
    print(cal.to_string(formatters={
        "predicted": "{:+.2f}".format,
        "actual":    "{:+.2f}".format,
        "resid":     "{:+.2f}".format,
        "home_win":  "{:.1%}".format,
    }))


def ats_by_edge(df, label, edge_col):
    section(f"{label} ATS by |edge vs Vegas|")
    d = df[~df["push"]].copy()
    d["abs_edge"] = d[edge_col].abs()
    bins = [0, 1, 2, 3, 4, 5, 7, 10, np.inf]
    d["edge_bucket"] = pd.cut(d["abs_edge"], bins=bins, right=False)
    ats_col = f"{label.lower()}_ats_correct"
    by = d.groupby("edge_bucket", observed=True).agg(
        n=(ats_col, "size"),
        win_rate=(ats_col, "mean"),
    )
    by["vs_break_even"] = by["win_rate"] - 0.5238  # break-even at -110
    print(by.to_string(formatters={
        "win_rate":      "{:.1%}".format,
        "vs_break_even": "{:+.1%}".format,
    }))


def hottake_contrarian_test(df):
    """
    The hypothesis: when HOTTAKE diverges from REACT (HOTTAKE has team A more bullishly
    than REACT does), HOTTAKE is reflecting recency bias the market also has,
    so fading HOTTAKE = backing REACT = positive ATS edge.

    'HOTTAKE more bullish on home' = hottake_vs_react > 0.
    Fading HOTTAKE = backing the AWAY side relative to HOTTAKE's pick.
    """
    section("HOTTAKE contrarian test: when HOTTAKE diverges from REACT")
    d = df[~df["push"]].copy()
    d["divergence"] = d["hottake_vs_react"].abs()
    bins = [0, 1, 2, 3, 5, 7, np.inf]
    d["div_bucket"] = pd.cut(d["divergence"], bins=bins, right=False)

    # When HOTTAKE > REACT (HOTTAKE more bullish on home), did HOME cover (= HOTTAKE was "right")?
    # When HOTTAKE < REACT (HOTTAKE less bullish on home), did HOME cover (= HOTTAKE was "wrong")?
    # We'll compute: of the games where HOTTAKE picked DIFFERENTLY from REACT, what fraction did
    # HOTTAKE's pick win ATS vs REACT's pick winning ATS?
    diverge = d[d["react_picks_home"] != d["hottake_picks_home"]].copy()
    print(f"\n  Games where REACT and HOTTAKE pick OPPOSITE sides: {len(diverge)}")
    print(f"    HOTTAKE's pick wins ATS: {fmt_pct(diverge['hottake_ats_correct'].mean())}")
    print(f"    REACT's pick wins ATS:   {fmt_pct(diverge['react_ats_correct'].mean())}")
    # The "fade HOTTAKE" strategy = always take the REACT side when they diverge
    print(f"    => 'Fade HOTTAKE / back REACT' ATS: {fmt_pct(diverge['react_ats_correct'].mean())}")

    print("\n  ATS by divergence magnitude (HOTTAKE's pick):")
    print(d.groupby("div_bucket", observed=True).agg(
        n=("hottake_ats_correct","size"),
        HOTTAKE_ATS=("hottake_ats_correct","mean"),
        REACT_ATS=("react_ats_correct","mean"),
    ).to_string(formatters={
        "HOTTAKE_ATS": "{:.1%}".format,
        "REACT_ATS":   "{:.1%}".format,
    }))


def playoff_split(df):
    section("Regular season vs Playoffs")
    reg = df[df["game_type"] == "REG"]
    po  = df[df["game_type"] != "REG"]
    summary(reg, "Regular season")
    summary(po,  "Playoffs")

    section("ATS by playoff round")
    for gt in ["WC", "DIV", "CON", "SB"]:
        g = df[df["game_type"] == gt]
        gns = g[~g["push"]]
        if len(gns) == 0: continue
        ats_r = gns["react_ats_correct"].mean()
        ats_h = gns["hottake_ats_correct"].mean()
        su_r  = g["react_su_correct"].mean()
        su_h  = g["hottake_su_correct"].mean()
        print(f"  {gt}  n={len(g)}  REACT ATS {fmt_pct(ats_r)} / SU {fmt_pct(su_r)}  |  HOTTAKE ATS {fmt_pct(ats_h)} / SU {fmt_pct(su_h)}")


def biggest_edges(df):
    section("Biggest REACT edges vs Vegas (top 15)")
    d = df[~df["push"]].copy()
    d["abs_edge"] = d["react_edge"].abs()
    cols = ["season","week","game_type","home_name","away_name",
            "react_home_margin","hottake_home_margin","vegas_home_margin","actual_home_margin",
            "react_edge","react_ats_correct"]
    print(d.nlargest(15, "abs_edge")[cols].to_string(index=False, formatters={
        "react_home_margin":   "{:+.2f}".format,
        "hottake_home_margin": "{:+.2f}".format,
        "vegas_home_margin":   "{:+.2f}".format,
        "actual_home_margin":  "{:+.1f}".format,
        "react_edge":          "{:+.2f}".format,
    }))

    section("Biggest HOTTAKE-vs-REACT divergences (HOTTAKE bullish, REACT skeptical) (top 15)")
    d2 = df.copy()
    d2["abs_div"] = d2["hottake_vs_react"].abs()
    cols2 = ["season","week","game_type","home_name","away_name",
             "react_home_margin","hottake_home_margin","vegas_home_margin","actual_home_margin",
             "hottake_vs_react","react_ats_correct","hottake_ats_correct"]
    print(d2.nlargest(15, "abs_div")[cols2].to_string(index=False, formatters={
        "react_home_margin":   "{:+.2f}".format,
        "hottake_home_margin": "{:+.2f}".format,
        "vegas_home_margin":   "{:+.2f}".format,
        "actual_home_margin":  "{:+.1f}".format,
        "hottake_vs_react":    "{:+.2f}".format,
    }))


def main():
    df = pd.read_csv(DATA)
    df["season"] = df["season"].astype(int)
    section(f"DILLON predictive analysis  -  {len(df)} games, {df['season'].min()}-{df['season'].max()}")
    summary(df, "ALL GAMES")

    playoff_split(df)
    by_season(df)
    calibration(df, "REACT",   "react_home_margin")
    calibration(df, "HOTTAKE", "hottake_home_margin")
    ats_by_edge(df, "REACT",   "react_edge")
    ats_by_edge(df, "HOTTAKE", "hottake_edge")
    hottake_contrarian_test(df)
    biggest_edges(df)


if __name__ == "__main__":
    main()
