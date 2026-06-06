# DILLON Predictive Analysis: How Well Do REACT and HOTTAKE Predict NFL Outcomes?

**Date of analysis:** 2026-05-22
**Window:** 1999-2025 NFL seasons (7,257 regular-season + playoff games with closing spreads)
**Question:** Do DILLON's two ratings (REACT and HOTTAKE) have predictive value vs Vegas? Is HOTTAKE a contrarian signal? Does any composite beat either alone?

---

## The setup

DILLON publishes two ratings per team per week:

- **REACT**: stable, longer-window team-strength signal
- **HOTTAKE**: faster-moving, recency-weighted signal

Both have rating units of "points vs an average team," so a natural game prediction is:

```
predicted_home_margin = home_rating - away_rating + HCA
```

(HCA = 2.5 for normal games, 0 for neutral-site games like the Super Bowl.)

For each game we compared each rating's predicted margin against Vegas's closing spread and the actual final margin. Spreads sourced from nflverse `games.csv` (free, MIT-licensed, single CSV covering 1999-present).

The user's two hypotheses going in:

1. REACT should be a more predictive signal than HOTTAKE
2. HOTTAKE might actually be inversely correlated (recency-chasing tendencies that the market also has)

---

## Headline results

| Metric | Vegas | REACT | HOTTAKE |
|---|---|---|---|
| Margin RMSE | 13.19 | 13.90 | 14.77 |
| Margin MAE | 10.26 | 10.95 | 11.70 |
| Straight-up accuracy | (66.5% by Vegas's pick) | 64.0% | 62.4% |
| ATS vs Vegas | - | 50.6% | 49.3% |

**Hypothesis #1 (REACT > HOTTAKE): clearly confirmed.** REACT trails Vegas by ~0.7 RMSE; HOTTAKE trails by ~1.6. REACT outperforms HOTTAKE on every metric.

---

## The contrarian test

When REACT and HOTTAKE pick OPPOSITE sides against Vegas (1,532 games):

- HOTTAKE's pick wins ATS: **46.9%**
- REACT's pick wins ATS: **53.1%**

Directionally, HOTTAKE *is* a fade signal: 46.9% is a losing strategy. But the inverse - backing REACT when they disagree - sits at 53.1%, just barely above the 52.38% break-even at standard -110 juice.

### Statistical reality check on 53.1%

By point estimate: above break-even. By inference: cannot reject "this is random."

- Standard error: 1.27%
- 95% confidence interval: **[50.6%, 55.6%]**
- z-test vs 52.38% break-even: z=0.59, p=0.555
- Expected profit at 53.1%: ~$2,100 on $153K wagered
- Expected loss at the CI lower bound (50.6%): ~$3,500

We can't statistically distinguish this from a 50.5% strategy. You'd need ~5x the sample to even start the case for a real edge.

### When the divergence is large, HOTTAKE drops sharply

| HOTTAKE vs REACT divergence | n | HOTTAKE ATS | REACT ATS |
|---|---|---|---|
| 0-1 pts | 1,381 | 52.1% | 52.0% |
| 3-5 pts | 1,640 | 52.0% | 52.3% |
| 5-7 pts | 892 | 47.6% | 49.4% |
| **7+ pts** | **759** | **43.7%** | 50.3% |

At 7+ pt divergences, HOTTAKE is a 43.7% bet (real losing signal). But the *opposite* side (REACT) only hits 50.3% on the same games. So the edge is "HOTTAKE is wrong" rather than "REACT is right". Hypothesis #2 is *directionally* confirmed but doesn't generate a profitable strategy.

---

## Calibration: both ratings over-extrapolate, HOTTAKE much worse

**REACT calibration:**

| Predicted | Actual | Residual |
|---|---|---|
| under -14 | -8.6 | **+5.4** |
| 14+ | +11.9 | **-6.0** |

**HOTTAKE calibration:**

| Predicted | Actual | Residual |
|---|---|---|
| under -14 | -6.8 | **+11.4** |
| 14+ | +10.5 | **-9.0** |

When HOTTAKE predicts a 14+ point road favorite (n=402), the actual game margin averages -6.8 (so the favorite wins by ~7, not 14+). When it predicts a 14+ point home favorite (n=938), the actual is +10.5. HOTTAKE's hot teams are systematically not as hot as it thinks - this is the recency-bias signature directly visible in the data.

REACT has the same issue but milder. The MARGIN_CAP=35 in training only partially protects the prediction output.

---

## Playoffs (n=309)

The one place we found a meaningful signal.

| Cut | n | REACT ATS | HOTTAKE ATS |
|---|---|---|---|
| Regular season | 6,948 | 50.5% | 49.1% |
| **Playoffs (all)** | 309 | **53.8%** | 52.8% |
| Wild Card | 120 | 49.6% | 53.8% |
| **Divisional** | 108 | **58.5%** | 51.9% |
| Conference | 54 | 51.9% | 46.3% |
| Super Bowl | 27 | 57.7% | 65.4% |

**REACT in the Divisional round: 58.5% ATS on n=108.** That's a meaningful sample, well above break-even. Aligns with the broader thesis: in playoffs, load management and lineup noise drop out, the team-strength signal cleans up, and the market may not fully adjust.

(Same caveat about sample size applies - 108 games is small. But the directional finding lines up with the DUNCAN playoff result.)

---

## Composite signal sweep

Three ways to combine REACT and HOTTAKE were tested:

### 1. Convex combination

`predicted = w * REACT + (1-w) * HOTTAKE`. Best result: **w = 1.0 (pure REACT).** Adding any HOTTAKE strictly worsens RMSE and ATS. HOTTAKE contains no signal that REACT lacks.

### 2. Contrarian fade

`predicted = (1+α) * REACT - α * HOTTAKE`. A tiny contrarian weight (α=0.10-0.20) marginally improves RMSE (13.88 vs 13.90), but the improvement is noise-level. Aggressive contrarian (α=1+) wrecks RMSE in exchange for trivial ATS gains.

### 3. Unconstrained OLS

Best linear fit on the full 7,257 games:

```
actual_margin = -9.35 + 0.655 * REACT_predicted - 0.020 * HOTTAKE_predicted + 4.03 * HCA
```

Three findings here:

- **REACT coefficient is 0.655, not 1.0.** Optimal use of REACT is to *shrink* its predicted margin by ~35%. This is the same over-extrapolation problem the calibration table flags.
- **HOTTAKE coefficient is essentially zero** (-0.020). Given REACT in the model, HOTTAKE adds nothing.
- **Constant -9.35 + HCA correction +4.03** absorbs other systematic offsets.

This composite hits **RMSE 13.52 (vs REACT alone's 13.90)** - a real point-prediction improvement.

### Out-of-sample test (train 1999-2019, test 2020-2025)

| Predictor | RMSE | ATS |
|---|---|---|
| OLS composite | 13.15 | 49.8% |
| REACT only | 13.49 | 50.3% |

**The composite wins on margin RMSE but LOSES on ATS.** The composite shrinks predicted margins toward Vegas's spread, which improves point accuracy but doesn't pick different sides than REACT did. ATS performance is about which side of the spread you take, not how close your number is.

---

## Bottom line

1. **REACT is the genuinely predictive signal.** ~64% SU accuracy, RMSE within 0.7 of Vegas. Real signal for who wins games.
2. **HOTTAKE adds no predictive value once you have REACT.** Composite analysis shows its coefficient drops to essentially zero. As a standalone, it's a losing ATS bet (49.3%).
3. **HOTTAKE directionally is a contrarian indicator** at large REACT-HOTTAKE divergences (43.7% ATS at 7+ pt gap) - but backing the REACT side only hits 50.3% on those same games. The mispricing is "HOTTAKE wrong" not "REACT right."
4. **Best calibrated predicted margin** is `0.655 * REACT_diff + 4.66 * HCA` (the OLS form). Use this if you ever want to *show* a predicted spread on the site - it beats raw REACT on RMSE by ~0.4 pts.
5. **Nothing here beats Vegas ATS at any statistically significant threshold.** Same conclusion as the DUNCAN analysis.
6. **Playoff Divisional round is the only cell with a real-looking edge** (REACT 58.5% ATS, n=108). Could justify a playoff-specific feature.

---

## Site-feature implications

### Viable

1. **"Tonight's matchup preview"** showing REACT-predicted spread + SU favorite, framed as transparency. Use the calibrated 0.655 shrunk form for the displayed number.
2. **Playoff Predictions board** - REACT's 53.8% playoff ATS is honest content.
3. **Calibration page** showing how predicted vs actual line up across margin buckets.

### Not viable

- Anything framed as "beat the spread" or "fade the market."
- Selling HOTTAKE as a predictive signal. It's useful as a *vibe* or *form indicator*, not a forecast.
- A "follow DILLON to win money" feature.

---

## Artifacts

All in `NFL/predictive_analysis/`:

- `build_dataset.py` - joins DILLON games + REACT + HOTTAKE ratings + nflverse spreads
- `analyze.py` - per-rating accuracy, calibration, contrarian test, playoff splits
- `composite.py` - convex/contrarian/OLS composite sweep + out-of-sample test
- `dataset.csv` - 7,257 game rows
