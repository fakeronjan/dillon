"""Weekly Matchups: every game of a week, previewed from the ratings going
into it - win probability, DILLON's line, a projected score, and the stakes
(each team's playoff and Super Bowl odds with a win vs with a loss). After
the games: the final score and whether DILLON's pick was right.

Stakes come from the season sim (playoff_sim.py): one run of 100k simulations
from the snapshot before the week, split by each game's simulated result.

Line and projected score (fit on 1999-2025 games, leave-one-season-out):
    margin = LINE_LAM * (home rating - away rating + home edge)
        home edge = the sim's era home-field value (0 at neutral sites), so the
        line and the win probability always name the same favorite; LINE_LAM
        because ratings over-extrapolate big mismatches
    total  = 2 * league points per team-game last season
             + TOTAL_B * (both offenses - both defenses)
    projected score = (total +/- margin) / 2
The ratings don't beat betting lines (DILLON predictive
analysis, 2026-09-26: 50.6% against the spread).
"""
import numpy as np
import pandas as pd

import playoff_sim

N_SIMS = 100_000                  # fleet standard for settled odds
LINE_LAM = 0.63
TOTAL_B = 0.545


def _record(w, l, t):
    return f"{w}-{l}" + (f"-{t}" if t else "")


def build_season(season, games, ratings, conf_div, schedule=None, n_sims=N_SIMS, log=print):
    """games: all NFL games (season, week, week_id, home, away, home_pts,
    visitor_pts, is_neutral). ratings: (season, week_id, name, rating,
    rating_o, rating_d). Returns the season's weeks (list of dicts)."""
    g = games[games['season'] == season].copy()
    if schedule is not None and len(schedule):
        sch = schedule.copy()
        sch['season'] = season
        sch['week_id'] = np.nan
        sch['home_pts'] = np.nan
        sch['visitor_pts'] = np.nan
        g = pd.concat([g, sch[g.columns.intersection(sch.columns)]], ignore_index=True)
    prev = games[games['season'] == season - 1]
    mu = float(pd.concat([prev['home_pts'], prev['visitor_pts']]).mean()) if len(prev) else 21.5
    snap = {w: x.set_index('name') for w, x in ratings.groupby('week_id')}
    sim_ratings = {w: dict(zip(x.index, x['rating'])) for w, x in snap.items()}
    played_g = g[g['home_pts'].notna()]
    sim = playoff_sim.SeasonSim(season, played_g.dropna(subset=['week_id']), conf_div, sim_ratings,
                                schedule)
    first_week = 102 if playoff_sim.fmt(season) == 'four' else 101
    weeks_out = []
    all_ids = sorted(snap)
    for week, wg in g.groupby('week', sort=True):
        wid_game = wg['week_id'].min()
        if np.isnan(wid_game):                       # unplayed week: after the latest snapshot
            wid_prev = max(i for i in all_ids if i <= played_g['week_id'].max())
        else:
            earlier = [i for i in all_ids if i < wid_game]
            if not earlier:
                continue
            wid_prev = earlier[-1]
        if wid_prev not in sim.ratings:
            sim.ratings[wid_prev] = sim_ratings[wid_prev]
        cap = {}
        sim.odds_at(wid_prev, n_sims=n_sims, capture=cap)
        teams = cap['teams']
        idx = {t: i for i, t in enumerate(teams)}
        made, champ = cap['made'], cap['champ']
        rt = snap[wid_prev]
        # records going into the week (regular season)
        before = played_g[(played_g['week'] < min(week, 100))]
        rec = {}
        for h, a, hp, vp in before[['home', 'away', 'home_pts', 'visitor_pts']].itertuples(index=False):
            for t, pf, pa in ((h, hp, vp), (a, vp, hp)):
                w, l, tt = rec.get(t, (0, 0, 0))
                rec[t] = (w + (pf > pa), l + (pf < pa), tt + (pf == pa))
        games_out = []
        for x in wg.itertuples(index=False):
            h, a = x.home, x.away
            if h not in idx or a not in idx:
                continue
            ih, ia = idx[h], idx[a]
            stakes = None
            if week < 100:
                rest = cap.get('rest')
                j = rest.index[(rest['home'] == h) & (rest['away'] == a)] if rest is not None else []
                if not len(j):
                    continue
                hw = cap['hw'][:, j[0]]
                p_home = float(hw.mean())
                stakes = {}
                for t, won in ((h, hw), (a, ~hw)):
                    it = idx[t]
                    stakes[t] = {'po_win': float(made[won, it].mean()) if won.any() else None,
                                 'po_loss': float(made[~won, it].mean()) if (~won).any() else None,
                                 'sb_win': float((champ[won] == it).mean()) if won.any() else None,
                                 'sb_loss': float((champ[~won] == it).mean()) if (~won).any() else None}
            else:
                rnd = int(week) - first_week + 1
                hit = [c for c in cap['ps_games'] if c[0] == rnd and {c[1], c[2]} == {h, a}]
                if not hit:
                    continue
                _, ta, tb, won_a = hit[0]
                hw = won_a if ta == h else ~won_a
                p_home = float(hw.mean())
                stakes = {}
                for t, won in ((h, hw), (a, ~hw)):
                    it = idx[t]
                    stakes[t] = {'po_win': None, 'po_loss': None,
                                 'sb_win': float((champ[won] == it).mean()) if won.any() else None,
                                 'sb_loss': 0.0}
            neutral = int(getattr(x, 'is_neutral', 0) or 0) == 1
            rh, ra = rt.loc[h], rt.loc[a]
            margin = LINE_LAM * (rh['rating'] - ra['rating'] + (0.0 if neutral else sim.hp))
            total = 2 * mu + TOTAL_B * ((rh['rating_o'] + ra['rating_o']) - (rh['rating_d'] + ra['rating_d']))
            game = {
                'home': h, 'away': a, 'neutral': neutral,
                'home_record': _record(*rec.get(h, (0, 0, 0))), 'away_record': _record(*rec.get(a, (0, 0, 0))),
                'home_rating': round(float(rh['rating']), 2), 'away_rating': round(float(ra['rating']), 2),
                'p_home': round(p_home, 4),
                'line': round(float(margin) * 2) / 2,           # home by this many (negative = away)
                'proj_home': int(round((total + margin) / 2)), 'proj_away': int(round((total - margin) / 2)),
                'stakes': {k: {kk: (None if vv is None else round(vv, 4)) for kk, vv in v.items()} for k, v in stakes.items()},
                'result': None,
            }
            if not np.isnan(x.home_pts):
                game['result'] = {'home': int(x.home_pts), 'away': int(x.visitor_pts)}
            games_out.append(game)
        weeks_out.append({'week': int(week), 'games': games_out})
        log(f'  {season} week {week}: {len(games_out)} games')
    return weeks_out


# ── Juice: Quality x Stakes ──────────────────────────────────────────────────
# Quality = 60% the worse team's rating, 40% closeness (how near a toss-up);
# Stakes  = both teams' playoff-odds swing + K x their Super Bowl-odds swing,
#           K ramping 4 -> 8 over the regular season (seeding and title odds
#           matter more late); playoff games: Super Bowl swing only, K = 8.
# Each is ranked (0-100) against every game in the pool, then
# Juice = sqrt(Quality x Stakes): a game has to deliver on both.
SB_K_START, SB_K_END = 4.0, 8.0


def add_juice(seasons):
    """seasons: {season: weeks (build_season output)}. Adds 'quality',
    'stakes' and 'juice' (0-100) to every game, ranked against all of them."""
    rows = []
    for season, weeks in seasons.items():
        last = max((w['week'] for w in weeks if w['week'] < 100), default=18)
        for w in weeks:
            ps = w['week'] >= 100
            k = SB_K_END if ps else SB_K_START + (SB_K_END - SB_K_START) * (w['week'] - 1) / max(last - 1, 1)
            for g in w['games']:
                st = list(g['stakes'].values())
                po = 0.0 if ps else sum((s['po_win'] or 0) - (s['po_loss'] or 0) for s in st)
                sb = sum((s['sb_win'] or 0) - (s['sb_loss'] or 0) for s in st)
                rows.append((g, min(g['home_rating'], g['away_rating']), 1 - abs(2 * g['p_home'] - 1), po + k * sb))
    if not rows:
        return
    df = pd.DataFrame([r[1:] for r in rows], columns=['qmin', 'close', 'stake'])
    q = (0.6 * df['qmin'].rank(pct=True) + 0.4 * df['close'].rank(pct=True)).rank(pct=True) * 100
    s = df['stake'].rank(pct=True) * 100
    for (g, *_), qq, ss in zip(rows, q, s):
        g['quality'], g['stakes_score'] = int(round(qq)), int(round(ss))
        g['juice'] = int(round(np.sqrt(qq * ss)))
