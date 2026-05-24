"""
Test composite signals between REACT and HOTTAKE.

Three families:
  1. Convex combination: w * REACT + (1-w) * HOTTAKE
  2. Contrarian fade:    (1+a) * REACT - a * HOTTAKE  (= REACT + a*(REACT - HOTTAKE))
  3. Unconstrained linear regression of actual margin on REACT, HOTTAKE, HCA
"""

from pathlib import Path
import numpy as np
import pandas as pd

DATA = Path(__file__).resolve().parent / "dataset.csv"
BREAK_EVEN = 0.5238  # -110 juice


def metrics(predicted, actual, vegas, push_mask):
    """Return (RMSE, MAE, SU%, ATS%, n_bets) for a predicted-margin column."""
    err = predicted - actual
    rmse = np.sqrt(np.mean(err ** 2))
    mae  = np.mean(np.abs(err))
    su   = np.mean((predicted > 0) == (actual > 0))
    picks_home = predicted > vegas
    home_covers = actual > vegas
    correct = (picks_home & home_covers) | (~picks_home & ~home_covers)
    ats_mask = ~push_mask
    ats = correct[ats_mask].mean()
    return rmse, mae, su, ats, ats_mask.sum()


def main():
    df = pd.read_csv(DATA)
    actual = df["actual_home_margin"].values
    vegas  = df["vegas_home_margin"].values
    react  = df["react_home_margin"].values
    hot    = df["hottake_home_margin"].values
    push   = df["push"].values.astype(bool)
    hca    = df["hca"].values

    print("=" * 72)
    print(f"DILLON composite signal sweep  -  n={len(df)} games")
    print("=" * 72)

    # Baselines
    print("\nBaselines:")
    for name, pred in [("Vegas (predict spread)", vegas), ("REACT only", react), ("HOTTAKE only", hot)]:
        r, m, s, a, n = metrics(pred, actual, vegas, push)
        print(f"  {name:24s}  RMSE {r:.2f}  MAE {m:.2f}  SU {s:.1%}  ATS {a:.1%}  (n={n})")

    print("\n--- Convex combination: w * REACT + (1-w) * HOTTAKE ---")
    print(f"{'w':>5} {'RMSE':>6} {'MAE':>6} {'SU%':>6} {'ATS%':>6}  {'vs BE':>7}")
    for w in [0.0, 0.25, 0.5, 0.6, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1.0]:
        pred = w * react + (1 - w) * hot
        r, m, s, a, _ = metrics(pred, actual, vegas, push)
        print(f"{w:5.2f} {r:6.2f} {m:6.2f} {s:6.1%} {a:6.1%}  {a - BREAK_EVEN:+7.2%}")

    print("\n--- Contrarian fade: (1+a) * REACT - a * HOTTAKE  [a=0 is pure REACT] ---")
    print(f"{'a':>5} {'RMSE':>6} {'MAE':>6} {'SU%':>6} {'ATS%':>6}  {'vs BE':>7}")
    for a in [-0.5, -0.25, 0.0, 0.1, 0.2, 0.3, 0.5, 0.75, 1.0, 1.5, 2.0]:
        pred = (1 + a) * react - a * hot
        r, m, s, a_ats, _ = metrics(pred, actual, vegas, push)
        print(f"{a:5.2f} {r:6.2f} {m:6.2f} {s:6.1%} {a_ats:6.1%}  {a_ats - BREAK_EVEN:+7.2%}")

    print("\n--- Optimal weights via least-squares (overfit warning: in-sample) ---")
    # Fit actual = b0 + b1 * react + b2 * hottake + b3 * hca
    X = np.column_stack([np.ones(len(df)), react, hot, hca])
    coefs, *_ = np.linalg.lstsq(X, actual, rcond=None)
    b0, b1, b2, b3 = coefs
    print(f"  actual_margin ~= {b0:+.3f} + {b1:+.3f} * REACT_pred + {b2:+.3f} * HOTTAKE_pred + {b3:+.3f} * HCA")
    print(f"  (REACT_pred and HOTTAKE_pred already include each model's HCA, so b3 is incremental)")
    pred_ols = b0 + b1 * react + b2 * hot + b3 * hca
    r, m, s, a, _ = metrics(pred_ols, actual, vegas, push)
    print(f"  RMSE {r:.2f}  MAE {m:.2f}  SU {s:.1%}  ATS {a:.1%}  vs BE {a-BREAK_EVEN:+.2%}")

    # Fit using just the underlying rating diffs (no model-HCA double-count)
    print("\n--- OLS on raw rating diffs: actual ~ a*react_diff + b*hottake_diff + HCA ---")
    react_diff = df["home_react"].values - df["away_react"].values
    hot_diff   = df["home_hottake"].values - df["away_hottake"].values
    X2 = np.column_stack([np.ones(len(df)), react_diff, hot_diff, hca])
    coefs2, *_ = np.linalg.lstsq(X2, actual, rcond=None)
    c0, c1, c2, c3 = coefs2
    print(f"  actual_margin ~= {c0:+.3f} + {c1:+.3f} * (home_react - away_react) + {c2:+.3f} * (home_hot - away_hot) + {c3:+.3f} * HCA")
    pred_ols2 = c0 + c1 * react_diff + c2 * hot_diff + c3 * hca
    r, m, s, a, _ = metrics(pred_ols2, actual, vegas, push)
    print(f"  RMSE {r:.2f}  MAE {m:.2f}  SU {s:.1%}  ATS {a:.1%}  vs BE {a-BREAK_EVEN:+.2%}")

    # Out-of-sample test via simple time split
    print("\n--- Out-of-sample test: fit on 1999-2019, test on 2020-2025 ---")
    train = df["season"] <= 2019
    test  = df["season"] >= 2020
    X_train = np.column_stack([
        np.ones(train.sum()),
        react_diff[train], hot_diff[train], hca[train]
    ])
    coefs_t, *_ = np.linalg.lstsq(X_train, actual[train], rcond=None)
    d0, d1, d2, d3 = coefs_t
    print(f"  Train coefs: const={d0:+.3f}  REACT_diff={d1:+.3f}  HOT_diff={d2:+.3f}  HCA={d3:+.3f}")
    pred_test = d0 + d1 * react_diff[test] + d2 * hot_diff[test] + d3 * hca[test]
    r, m, s, a, n = metrics(pred_test, actual[test], vegas[test], push[test])
    print(f"  TEST set (n={test.sum()}):  RMSE {r:.2f}  MAE {m:.2f}  SU {s:.1%}  ATS {a:.1%}  vs BE {a-BREAK_EVEN:+.2%}  (n_bets={n})")
    # Compare to REACT-only on the same test set
    r2, m2, s2, a2, n2 = metrics(react[test], actual[test], vegas[test], push[test])
    print(f"  REACT-only on TEST:        RMSE {r2:.2f}  MAE {m2:.2f}  SU {s2:.1%}  ATS {a2:.1%}  vs BE {a2-BREAK_EVEN:+.2%}")

    # ATS test on the divergence-disagreement subset for the contrarian models
    print("\n--- Contrarian-fade ATS performance restricted to REACT/HOTTAKE disagreement ---")
    disagree = (df["react_picks_home"].values != df["hottake_picks_home"].values) & ~push
    print(f"  Disagreement n={disagree.sum()}")
    for a_param in [0.0, 0.25, 0.5, 1.0, 1.5, 2.0]:
        pred = (1 + a_param) * react - a_param * hot
        picks_home = pred > vegas
        home_covers = actual > vegas
        correct = (picks_home & home_covers) | (~picks_home & ~home_covers)
        ats = correct[disagree].mean()
        # CI: standard error of proportion
        n = disagree.sum()
        se = np.sqrt(ats * (1 - ats) / n)
        z = (ats - BREAK_EVEN) / se
        print(f"  a={a_param:5.2f}  ATS {ats:.1%}  SE {se:.2%}  z vs BE = {z:+.2f}  (p={2*(1-_norm_cdf(abs(z))):.3f})")


def _norm_cdf(x):
    """Approximate normal CDF without scipy."""
    return 0.5 * (1 + np.tanh(0.7978845608 * x * (1 + 0.044715 * x * x)))


if __name__ == "__main__":
    main()
