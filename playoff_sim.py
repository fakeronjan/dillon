"""DILLON title odds: Monte Carlo of the rest of the NFL season + playoffs.

Port of LOBO/DUNCAN/GRIFFEY's playoff_sim.py. At every weekly snapshot,
simulate the remaining regular-season games, qualify and seed each
conference under that season's format, then play the bracket. Anything
already played is fixed. Snapshots are keyed by DILLON's week id
(cume_week_id / ranking_id), not dates.

Game model (probit on 12,618 NFL games 1970-2026, pre-game snapshot
ratings, fit by log loss per decade):
    P(home win) = Phi(A * (rating_home - rating_away + home_pts))
Home field shrank from ~5.5 pts (1990s) to ~1.9 (2020s). Super Bowls (and
other neutral-site games) have no home edge. Regular-season ties count as
half a win.
"""
import json as _json
import os as _os

import numpy as np
import pandas as pd
from scipy.special import ndtr

N_SIMS = 10_000            # fleet standard: regular-season snapshots
N_SIMS_PLAYOFFS = 100_000  # once the regular season is over

ERA_PARAMS = [             # (first season, A, home pts)
    (1970, 0.0584, 3.83),
    (1980, 0.0504, 4.41),
    (1990, 0.0498, 5.50),
    (2000, 0.0491, 4.17),
    (2010, 0.0521, 3.52),
    (2020, 0.0511, 1.92),
]

# Ratings aren't fixed for the rest of the season: a week-2 rating is mostly
# last year's team. Each simulation gives every team a random offset for the
# remaining games, SD = drift_sd(share of regular season left): the rating
# error that best explains how the rest of each 1972-2025 season actually
# went, given that week's ratings. (Fitting how far ratings later MOVE, as
# before, understated early error by half and kept drift going past midseason
# when the ratings had already caught up.)
DRIFT_LEFT = [0.0, 0.43, 0.59, 0.72, 0.88, 1.0]
DRIFT_SD = [0.0, 0.0, 2.97, 7.45, 11.88, 14.44]


def drift_sd(frac_left):
    return float(np.interp(frac_left, DRIFT_LEFT, DRIFT_SD))

_TB = _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), 'nfl_tiebreak_orders.json')
# Standings ties the NFL broke differently from our simplified tiebreak
# (head-to-head, then point differential). Per season, tied teams in the
# order they were actually seeded; earlier wins. From search_tiebreaks.py.
TIEBREAK_WINNERS = ({int(k): v for k, v in _json.load(open(_TB)).items()}
                    if _os.path.exists(_TB) else {})


_WC = _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), 'nfl_wc_opponents.json')
# Early divisional-round matchups followed a rotation, not seeding: the wild
# card's actual first opponent per season/conference, where it broke the
# usual rule (top division winner unless division rivals). From
# search_tiebreaks.py alongside the tie orders.
WC_OPPONENT = ({int(k): v for k, v in _json.load(open(_WC)).items()}
               if _os.path.exists(_WC) else {})


def era_params(season):
    a, h = ERA_PARAMS[0][1:]
    for start, aa, hh in ERA_PARAMS:
        if season >= start:
            a, h = aa, hh
    return a, h


def fmt(season):
    if season == 1982:
        return 'strike'     # 8 per conference by record, divisions ignored
    if season <= 1977:
        return 'four'       # 3 division winners + 1 wild card
    if season <= 1989:
        return 'five'       # + wild card game between two wild cards
    if season <= 2019:
        return 'six'        # 12 teams, seeds 1-2 bye, re-seeded
    return 'seven'          # 14 teams, seed 1 byes, re-seeded


def n_seeds(season):
    return {'strike': 8, 'four': 4, 'five': 5, 'six': 6, 'seven': 7}[fmt(season)]


def round_names(season):
    """(full, short) names for every round, first to last."""
    if fmt(season) == 'four':
        return (['Divisional Round', 'Conference Championship', 'Super Bowl'], ['DIV', 'CONF', 'SB'])
    return (['Wild Card Round', 'Divisional Round', 'Conference Championship', 'Super Bowl'],
            ['WC', 'DIV', 'CONF', 'SB'])


def entry_rounds(season):
    """Seed label ('A1', 'N6', ...) -> round that seed enters (later = bye)."""
    f = fmt(season)
    out = {}
    for c in 'AN':
        for k in range(1, n_seeds(season) + 1):
            if f == 'five':
                r = 1 if k >= 4 else 2
            elif f == 'six':
                r = 1 if k >= 3 else 2
            elif f == 'seven':
                r = 1 if k >= 2 else 2
            else:
                r = 1
            out[f'{c}{k}'] = r
    return out


class SeasonSim:
    def __init__(self, season, games, conf_div, ratings, schedule=None):
        """games: the season's games (week_id = DILLON cume_week_id, week,
        home, away, home_pts, visitor_pts, is_neutral). Weeks >= 100 are the
        playoffs. schedule: unplayed regular-season games (home, away)."""
        self.season = season
        g = games.copy()
        rs = g[g['week'] < 100][['week_id', 'home', 'away', 'home_pts', 'visitor_pts', 'is_neutral']]
        if schedule is not None and len(schedule):
            sch = schedule.assign(week_id=np.inf, home_pts=np.nan, visitor_pts=np.nan)
            if 'is_neutral' not in sch.columns:
                sch['is_neutral'] = 0
            rs = pd.concat([rs, sch[rs.columns]], ignore_index=True)
        # International / neutral-site regular-season games get no home edge.
        rs['is_neutral'] = rs['is_neutral'].fillna(0).astype(int)
        self.teams = sorted(set(rs['home']) | set(rs['away']))
        self.idx = {t: i for i, t in enumerate(self.teams)}
        cd = [conf_div(t, season) for t in self.teams]
        self.conf = np.array([c for c, _ in cd])
        self.div = np.array([c + ' ' + d for c, d in cd])
        self.A, self.hp = era_params(season)
        self.rs = rs.assign(h=rs['home'].map(self.idx), a=rs['away'].map(self.idx))
        ps = g[g['week'] >= 100].copy()
        ps['winner'] = np.where(ps['home_pts'] > ps['visitor_pts'], ps['home'], ps['away'])
        self.ps = ps.sort_values('week_id', kind='stable')
        self.ratings = ratings
        self.n_rounds = len(round_names(season)[0])

    def rs_over(self, wid):
        return not ((self.rs['week_id'] > wid) | self.rs['home_pts'].isna()).any()

    def _standings(self, sub):
        T = len(self.teams)
        w = np.zeros(T); gp = np.zeros(T); pdiff = np.zeros(T)
        for h, a, hp, vp in sub[['h', 'a', 'home_pts', 'visitor_pts']].itertuples(index=False):
            gp[h] += 1; gp[a] += 1; pdiff[h] += hp - vp; pdiff[a] += vp - hp
            if hp > vp: w[h] += 1
            elif vp > hp: w[a] += 1
            else: w[h] += 0.5; w[a] += 0.5
        return w, gp, pdiff

    def _static_tiebreak(self, done):
        """Once the regular season is over, within each group tied on win%:
        head-to-head, then division record (when the tied teams share a
        division), then conference record, then point differential - a
        subset of the NFL's procedure. Recorded outcomes override (history
        is exact; this matters for the live season before the playoffs).
        Higher = better; only compared within a tie group."""
        T = len(self.teams)
        w, gp, pdiff = self._standings(done)
        pct = w / np.maximum(gp, 1)
        same_div = done[self.div[done['h']] == self.div[done['a']]]
        same_conf = done[self.conf[done['h']] == self.conf[done['a']]]
        dw, dg, _ = self._standings(same_div)
        cw, cg, _ = self._standings(same_conf)
        dpct = dw / np.maximum(dg, 1); cpct = cw / np.maximum(cg, 1)
        score = np.zeros(T)
        for p in np.unique(pct):
            grp = np.where(pct == p)[0]
            if len(grp) < 2:
                continue
            gs = set(grp)
            sub = done[done['h'].isin(gs) & done['a'].isin(gs)]
            hw, hg, _ = self._standings(sub)
            h2h = np.where(hg > 0, hw / np.maximum(hg, 1), 0.5)
            one_div = len(set(self.div[grp])) == 1
            key = sorted(grp, key=lambda t: (h2h[t], dpct[t] if one_div else 0, cpct[t], pdiff[t]),
                         reverse=True)
            for rank, t in enumerate(key):
                score[t] = len(key) - rank
        order = TIEBREAK_WINNERS.get(self.season, [])
        for k, t in enumerate(order):
            if t in self.idx:
                score[self.idx[t]] += 1e6 * (len(order) - k)
        return score

    def odds_at(self, wid, n_sims=N_SIMS, capture=None):
        """capture: optional dict, filled with per-simulation results for the
        Weekly Matchups tab (remaining games' home wins, playoff games still
        to play, who made the playoffs / won it all)."""
        season, f = self.season, fmt(self.season)
        T = len(self.teams)
        rng = np.random.default_rng(int(season * 1000 + (wid % 1000)))
        rt = self.ratings.get(wid, {})
        R = np.array([rt.get(t, 0.0) for t in self.teams])
        A, hp = self.A, self.hp

        played = self.rs['home_pts'].notna() & (self.rs['week_id'] <= wid)
        done, rest = self.rs[played], self.rs[~played]
        frac_left = len(rest) / max(len(self.rs), 1)
        sd = drift_sd(frac_left)
        Rs = R[None, :] + rng.normal(0.0, sd, (n_sims, T)) if sd > 0 else R  # per-sim strength
        w0, g0, _ = self._standings(done)
        W = np.tile(w0, (n_sims, 1)); G = np.tile(g0, (n_sims, 1))
        if len(rest):
            h = rest['h'].to_numpy(); a = rest['a'].to_numpy()
            hp_g = np.where(rest['is_neutral'].to_numpy() == 1, 0.0, hp)
            pr = ndtr(A * (Rs[:, h] - Rs[:, a] + hp_g)) if np.ndim(Rs) == 2 else ndtr(A * (R[h] - R[a] + hp_g))
            hw = (rng.random((n_sims, len(rest))) < pr).astype(np.float32)
            Hm = np.zeros((len(rest), T), np.float32); Hm[np.arange(len(rest)), h] = 1
            Am = np.zeros((len(rest), T), np.float32); Am[np.arange(len(rest)), a] = 1
            W += hw @ Hm + (1 - hw) @ Am
            G += (Hm + Am).sum(0)
            if capture is not None:
                capture['rest'] = rest.reset_index(drop=True)
                capture['hw'] = hw.astype(bool)
        pct = W / np.maximum(G, 1)
        static = self._static_tiebreak(done) if rest.empty else np.zeros(T)
        noise = rng.random((n_sims, T))
        sim_ix = np.arange(n_sims)
        S = 1 if rest.empty else n_sims        # final table: seed once, broadcast
        srank = np.unique(static, return_inverse=True)[1].astype(float)
        tie_term = (srank[None, :] + noise[:S]) * 1e-8
        pct_s = pct[:S]
        six = np.arange(S)

        def ranked(members, bonus=None):
            m = np.array(members)
            key = pct_s[:, m] + tie_term[:, m]
            if bonus is not None:
                key = key + bonus * 10.0
            return m[np.argsort(-key, axis=1, kind='stable')]

        seeds = {}
        for c in ('AFC', 'NFC'):
            m = np.where(self.conf == c)[0]
            if f == 'strike':
                seeds[c] = ranked(m)[:, :8]
                continue
            order = ranked(m)
            pos = np.empty((S, T), dtype=int)
            pos[six[:, None], order] = np.arange(len(m))[None, :]
            is_dw = np.zeros((S, T), bool)
            for dv in np.unique(self.div[m]):
                mem = m[self.div[m] == dv]
                is_dw[six, mem[np.argmin(pos[:, mem], axis=1)]] = True
            seeds[c] = ranked(m, is_dw[:, m].astype(float))[:, :n_seeds(season)]
        if S == 1:
            seeds = {k: np.broadcast_to(v, (n_sims, v.shape[1])) for k, v in seeds.items()}

        # Real games keyed by (pair, round): keying on the pair alone let a
        # wrong bracket "consume" a real game from a different round when the
        # same two teams met in both (1978 AFC: Miami-Houston was the wild
        # card game, New England-Houston the divisional game).
        first_week = 102 if f == 'four' else 101
        ps_by_pair = {}
        for r in self.ps[self.ps['week_id'] <= wid].itertuples(index=False):
            ps_by_pair.setdefault((frozenset((r.home, r.away)), int(r.week) - first_week + 1), []).append(r.winner)

        self.used_actual = 0
        self.rs_complete = rest.empty
        self.seeds = {}
        if self.rs_complete:
            for c, arr in seeds.items():
                for k, t in enumerate(arr[0]):
                    self.seeds[self.teams[t]] = f"{c[0]}{k + 1}"
        self.matchups = []
        if capture is not None:
            capture['ps_games'] = []
        reach = np.zeros((self.n_rounds + 2, T))
        entered = np.zeros((n_sims, T), dtype=bool)

        def play(a, sa, b, sb, rnd, neutral=False):
            """Single game; a/b team arrays, sa/sb seed numbers (lower hosts)."""
            for t in (a, b):
                new = ~entered[sim_ix, t]
                entered[sim_ix, t] = True
                np.add.at(reach[0], t[new], 1)
                for k in range(2, rnd):
                    np.add.at(reach[k], t[new], 1)
                np.add.at(reach[rnd], t, 1)
            fixed = np.all(a == a[0]) and np.all(b == b[0])
            actual = ps_by_pair.get((frozenset((self.teams[a[0]], self.teams[b[0]])), rnd), []) if fixed else []
            if actual:
                won = np.full(n_sims, actual[0] == self.teams[a[0]]); self.used_actual += 1
            else:
                edge = 0.0 if neutral else np.where(sa < sb, hp, -hp)
                ra = Rs[sim_ix, a] if np.ndim(Rs) == 2 else R[a]
                rb = Rs[sim_ix, b] if np.ndim(Rs) == 2 else R[b]
                won = rng.random(n_sims) < ndtr(A * (ra - rb + edge))
                if capture is not None and fixed:
                    capture['ps_games'].append((rnd, self.teams[a[0]], self.teams[b[0]], won))
            if fixed and self.rs_complete:
                ta, tb = self.teams[a[0]], self.teams[b[0]]
                self.matchups.append((rnd, 1, ta, tb, actual[:1], actual[0] if actual else None))
            return np.where(won, a, b), np.where(won, sa, sb)

        def reseed_pair(teams_, seeds_):
            """Best remaining seed vs worst, the other two meet."""
            tt = np.stack(teams_, 1); ss = np.stack(seeds_, 1)
            o = np.argsort(ss, axis=1)
            tt = np.take_along_axis(tt, o, 1); ss = np.take_along_axis(ss, o, 1)
            return (tt[:, 0], ss[:, 0], tt[:, -1], ss[:, -1]), (tt[:, 1], ss[:, 1], tt[:, 2], ss[:, 2])

        champs = {}
        for c in ('AFC', 'NFC'):
            sd = seeds[c]
            t = lambda k: sd[:, k - 1]
            n = lambda k: np.full(n_sims, k)
            if f == 'four':
                # the wild card plays the top division winner unless they're
                # division rivals, in which case it plays the second.
                wc = t(4)
                same = self.div[wc] == self.div[t(1)]
                forced = WC_OPPONENT.get(season, {}).get(c)
                if forced in self.idx:
                    hit = np.stack([t(1), t(2), t(3)], 1) == self.idx[forced]
                    k = np.where(hit.any(1), hit.argmax(1), np.where(same, 1, 0))
                else:
                    k = np.where(same, 1, 0)
                dws = np.stack([t(1), t(2), t(3)], 1)
                top, top_s = dws[sim_ix, k], k + 1
                ok = np.sort(np.stack([(k + 1) % 3, (k + 2) % 3], 1), axis=1)
                o1, o1_s = dws[sim_ix, ok[:, 0]], ok[:, 0] + 1
                o2, o2_s = dws[sim_ix, ok[:, 1]], ok[:, 1] + 1
                x = play(top, top_s, wc, n(4), 1)
                y = play(o1, o1_s, o2, o2_s, 1)
                champs[c] = play(*x, *y, 2)
            elif f == 'five':
                wc = play(t(4), n(4), t(5), n(5), 1)
                same = self.div[wc[0]] == self.div[t(1)]
                forced = WC_OPPONENT.get(season, {}).get(c)
                if forced in self.idx:
                    hit = np.stack([t(1), t(2), t(3)], 1) == self.idx[forced]
                    k = np.where(hit.any(1), hit.argmax(1), np.where(same, 1, 0))
                else:
                    k = np.where(same, 1, 0)
                dws = np.stack([t(1), t(2), t(3)], 1)
                top, top_s = dws[sim_ix, k], k + 1
                ok = np.sort(np.stack([(k + 1) % 3, (k + 2) % 3], 1), axis=1)
                o1, o1_s = dws[sim_ix, ok[:, 0]], ok[:, 0] + 1
                o2, o2_s = dws[sim_ix, ok[:, 1]], ok[:, 1] + 1
                x = play(top, top_s, wc[0], wc[1], 2)
                y = play(o1, o1_s, o2, o2_s, 2)
                champs[c] = play(*x, *y, 3)
            elif f == 'strike':
                r1 = [play(t(1), n(1), t(8), n(8), 1), play(t(2), n(2), t(7), n(7), 1),
                      play(t(3), n(3), t(6), n(6), 1), play(t(4), n(4), t(5), n(5), 1)]
                p1, p2 = reseed_pair([x[0] for x in r1], [x[1] for x in r1])
                x = play(*p1, 2); y = play(*p2, 2)
                champs[c] = play(*x, *y, 3)
            elif f == 'six':
                a36 = play(t(3), n(3), t(6), n(6), 1)
                a45 = play(t(4), n(4), t(5), n(5), 1)
                p1, p2 = reseed_pair([t(1), t(2), a36[0], a45[0]], [n(1), n(2), a36[1], a45[1]])
                x = play(*p1, 2); y = play(*p2, 2)
                champs[c] = play(*x, *y, 3)
            else:  # seven
                a27 = play(t(2), n(2), t(7), n(7), 1)
                a36 = play(t(3), n(3), t(6), n(6), 1)
                a45 = play(t(4), n(4), t(5), n(5), 1)
                p1, p2 = reseed_pair([t(1), a27[0], a36[0], a45[0]], [n(1), a27[1], a36[1], a45[1]])
                x = play(*p1, 2); y = play(*p2, 2)
                champs[c] = play(*x, *y, 3)
        (ea, es), (na, ns) = champs['AFC'], champs['NFC']
        champ, _ = play(ea, es, na, ns, self.n_rounds, neutral=True)
        np.add.at(reach[-1], champ, 1)
        if capture is not None:
            capture['made'] = entered.copy()
            capture['champ'] = champ
            capture['teams'] = self.teams
        reach /= n_sims
        cols = ['playoffs'] + [f'r{k}' for k in range(2, self.n_rounds + 1)] + ['champ']
        rows = np.vstack([reach[0]] + [reach[k] for k in range(2, self.n_rounds + 1)] + [reach[-1]])
        return pd.DataFrame(rows.T, index=self.teams, columns=cols)


def compute(games, ratings_df, conf_div, current_season, schedule=None, log=print):
    """games: (season, week, week_id, home, away, home_pts, visitor_pts,
    is_neutral). ratings_df: (season, week_id, name, rating). Returns (odds,
    brackets) keyed by week_id, like the other sites' (keyed by date)."""
    out, brackets = [], {}
    for season, g in games.groupby('season'):
        season = int(season)
        rsub = ratings_df[ratings_df['season'] == season]
        ratings = {w: dict(zip(x['name'], x['rating'])) for w, x in rsub.groupby('week_id')}
        if not ratings:
            continue
        sim = SeasonSim(season, g, conf_div, ratings, schedule if season == current_season else None)
        for wid in sorted(ratings):
            n = N_SIMS_PLAYOFFS if sim.rs_over(wid) else N_SIMS
            o = sim.odds_at(wid, n_sims=n)
            if sim.rs_complete:
                brackets.setdefault(season, {})[wid] = (dict(sim.seeds), list(sim.matchups), n)
            o.index.name = 'team'
            o = o.reset_index()
            o['season'] = season
            o['week_id'] = wid
            out.append(o)
        log(f"  {season}: {len(ratings)} snapshots")
    return pd.concat(out, ignore_index=True), brackets
