"""Remnant-only steps (no experimental outcomes):
  1. Winsorisation bounds (1st/99th percentiles) for every feature, from remnant units.
  2. Feature diagnostics: how much of each trajectory feature is linearly explained by level features.
  3. Prognostic score: gradient-boosted classifier of assignment completion trained on remnant units
     only (class-grouped CV for honest remnant performance), then applied to experimental units.

Outputs: data/derived/scores_exp.parquet, reports/audit_tables/winsor_bounds.json,
         reports/audit_tables/remnant_prognostic_cv.json, reports/audit_tables/traj_vs_level_r2.csv
"""
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LinearRegression
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score
from sklearn.model_selection import GroupKFold

from common import AUDIT, DERIVED, SEED, make_logger, save_json
import features as F

log = make_logger("03_remnant_score")
FEATS = F.LEVEL_FEATURES + F.TRAJECTORY_FEATURES


def winsorize(df, bounds):
    df = df.copy()
    for c, (lo, hi) in bounds.items():
        df[c] = df[c].clip(lo, hi)
    return df


def main():
    rem = pd.read_parquet(DERIVED / "feat_rem_b24_k20.parquet")
    log(f"remnant units: {len(rem):,}")
    bounds = {c: [float(np.nanquantile(rem[c], 0.01)), float(np.nanquantile(rem[c], 0.99))] for c in FEATS}
    save_json(bounds, AUDIT / "winsor_bounds.json")
    rem = winsorize(rem, bounds)

    # Trajectory features explained by level features (remnant units with >= 10 prior stats rows).
    r10 = rem[rem.n_prior_stats >= 10]
    Xb = r10[F.LEVEL_FEATURES].fillna(r10[F.LEVEL_FEATURES].median())
    rows = []
    for t in F.TRAJECTORY_FEATURES:
        ok = r10[t].notna()
        r2 = LinearRegression().fit(Xb[ok], r10.loc[ok, t]).score(Xb[ok], r10.loc[ok, t])
        rows.append({"feature": t, "r2_on_level_features": r2, "missing_share": float(1 - ok.mean()),
                     "sd": float(r10[t].std())})
    pd.DataFrame(rows).to_csv(AUDIT / "traj_vs_level_r2.csv", index=False)
    r10[FEATS].corr().round(3).to_csv(AUDIT / "remnant_feature_correlations.csv")

    # Prognostic model (all remnant units; NaN handled natively).
    X, y, g = rem[FEATS].to_numpy(float), rem.assignment_completed.to_numpy(int), rem.class_id.to_numpy()
    params = dict(max_iter=400, learning_rate=0.05, max_leaf_nodes=31, min_samples_leaf=100,
                  l2_regularization=1.0, early_stopping=True, validation_fraction=0.1, random_state=SEED)
    oof = np.zeros(len(y))
    for tr, te in GroupKFold(n_splits=5).split(X, y, g):
        m = HistGradientBoostingClassifier(**params).fit(X[tr], y[tr])
        oof[te] = m.predict_proba(X[te])[:, 1]
    cv = {"n": int(len(y)), "base_rate": float(y.mean()), "auc": float(roc_auc_score(y, oof)),
          "brier": float(brier_score_loss(y, oof)), "log_loss": float(log_loss(y, oof)),
          "auc_units_with_10_stats": float(roc_auc_score(y[rem.n_prior_stats >= 10], oof[rem.n_prior_stats >= 10]))}
    log(f"remnant class-grouped CV: {cv}")
    save_json(cv, AUDIT / "remnant_prognostic_cv.json")
    final = HistGradientBoostingClassifier(**params).fit(X, y)

    scores = []
    for variant in ("b24_k20", "b24_k10", "b0_k20"):
        fe = winsorize(pd.read_parquet(DERIVED / f"feat_exp_{variant}.parquet"), bounds)
        pr = np.clip(final.predict_proba(fe[FEATS].to_numpy(float))[:, 1], 1e-4, 1 - 1e-4)
        scores.append(pd.DataFrame({"unit_id": fe.unit_id, "variant": variant, "s": np.log(pr / (1 - pr))}))
    pd.concat(scores).to_parquet(DERIVED / "scores_exp.parquet", index=False)
    log("scores written")


if __name__ == "__main__":
    main()
