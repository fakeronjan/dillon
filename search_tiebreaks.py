"""One-off: recover how past NFL standings ties (and 1970s rotation-era
wild-card matchups) actually went.

For each season whose simulated bracket doesn't reproduce the real playoff
games, try every ordering of teams tied on win% at the end of the regular
season (plausible playoff teams only) and, in the 1970-89 formats, every
wild-card opponent; keep the first combination whose bracket consumes every
real playoff game and crowns the real champion. Writes
nfl_tiebreak_orders.json and nfl_wc_opponents.json (read by playoff_sim).
"""
import itertools, json, sys
import numpy as np
exec(open(sys.argv[1]).read().split("if __name__ == '__main__':")[0])
tb_out = dict(PS.TIEBREAK_WINNERS); wc_out = dict(PS.WC_OPPONENT)
for s_ in [int(x) for x in sys.argv[2].split(',')]:   # re-derive these seasons from scratch
    tb_out.pop(s_, None); wc_out.pop(s_, None); PS.TIEBREAK_WINNERS.pop(s_, None); PS.WC_OPPONENT.pop(s_, None)
for s in [int(x) for x in sys.argv[2].split(',')]:
    rsub = r[r.season == s]
    ratings = {w: dict(zip(x.name, x.rating)) for w, x in rsub.groupby('week_id')}
    sim = PS.SeasonSim(s, g[g.season == s], conf_div, ratings)
    last = sorted(ratings)[-1]
    ps_weeks = sorted(w for w in ratings if w >= sim.ps.week_id.min())
    w_, gp_, _ = sim._standings(sim.rs)
    pct = w_ / gp_
    groups = []
    for c in ('AFC', 'NFC'):
        m = np.where(sim.conf == c)[0]
        cut = np.sort(pct[m])[::-1][min(len(m) - 1, 9)]
        for p in np.unique(pct[m]):
            grp = [sim.teams[i] for i in m if pct[i] == p]
            if len(grp) > 1 and p >= cut:
                groups.append(grp)
    tie_opts = [list(itertools.permutations(grp)) for grp in groups]
    wc_opts = [None]
    if PS.fmt(s) in ('four', 'five'):
        confs = [[None] + [sim.teams[i] for i in np.where(sim.conf == c)[0]] for c in ('AFC', 'NFC')]
        wc_opts = [(a, b) for a in confs[0] for b in confs[1]]
    found, tried = None, 0
    for combo in itertools.product(*tie_opts):
        PS.TIEBREAK_WINNERS[s] = [t for grp in combo for t in grp]
        for wc in wc_opts:
            PS.WC_OPPONENT.pop(s, None)
            if wc:
                forced = {k: v for k, v in zip(('AFC', 'NFC'), wc) if v}
                if forced:
                    PS.WC_OPPONENT[s] = forced
            tried += 1
            o = sim.odds_at(last, n_sims=20)
            ok = sim.used_actual == len(sim.ps) and o.champ.max() == 1
            # every playoff snapshot, not just the last one
            for w in (w for w in ps_weeks if ok and w != last):
                sim.odds_at(w, n_sims=20)
                ok = sim.used_actual == int((sim.ps.week_id <= w).sum())
            if ok:
                found = (list(PS.TIEBREAK_WINNERS[s]), dict(PS.WC_OPPONENT.get(s, {})))
                break
        if found or tried > 200000:
            break
    PS.TIEBREAK_WINNERS.pop(s, None); PS.WC_OPPONENT.pop(s, None)
    if found:
        tb_out[s] = found[0]
        if found[1]:
            wc_out[s] = found[1]
    print(f"{s}: {len(groups)} tie groups, tried {tried} -> {'found' if found else 'NONE'}"
          + (f" wc={found[1]}" if found and found[1] else ''), flush=True)
json.dump({str(k): v for k, v in sorted(tb_out.items())}, open('nfl_tiebreak_orders.json', 'w'), indent=1)
json.dump({str(k): v for k, v in sorted(wc_out.items())}, open('nfl_wc_opponents.json', 'w'), indent=1)
