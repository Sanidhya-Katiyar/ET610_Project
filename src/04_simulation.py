"""Outcome-blind simulation: real features, real contrasts, real randomized arms and fold structure;
SYNTHETIC outcomes only. Evaluates bias, interval coverage, type-I error and power of the primary
Delta_R test (MB vs MBT) and the secondary tests, and calibrates the SESOI reference distribution.

All injected effect sizes are ASSUMPTIONS, chosen before any experimental outcome was examined.

Data-generating process (per randomized unit i in experiment e, arm a):
  P(Y=1) = clip( expit(s_i + delta_e) + theta_{e,a} + kappa_{e,a} * h_i , 0.01, 0.99 )
  s_i: remnant prognostic logit (real); delta_e ~ N(0, 0.3^2); theta_{e,a} ~ N(0, 0.02^2) arm main effects;
  h_i: standardized moderator index; kappa: arm-specific moderation loadings (scenario-specific).
  Contrast CATE for (a0, a1) = difference of the two clipped arm probabilities.

Scenarios (h = T index unless stated; j = arm position 0..k-1 in sorted label order):
  a_null       kappa = 0
  b_level      kappa = 0.03 * j, h = level index (standardized b_correct_norm_win)
  c_traj_small kappa = 0.015 * j
  d_traj_sesoi kappa = 0.03 * j            (SESOI: >= 3 pp per SD of the trajectory index)
  e_traj_mixed kappa ~ N(0, (0.03/sqrt2)^2) independently per arm (random directions)
  f_traj_large kappa = 0.06 * j
T index = standardized residual of z(t_slope_correct_norm) + z(t_recent_shift_correct) after linear
regression on the level features (i.e. trajectory information beyond level).

Usage: python src/04_simulation.py [--reps 100] [--jobs 12]
"""
import argparse
import time

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from common import AUDIT, SEED, make_logger, save_json
import data_access as D
import pipeline as P

SCEN = ["a_null", "b_level", "c_traj_small", "d_traj_sesoi", "e_traj_mixed", "f_traj_large"]


def indices(df):
    X = df[D.XB].fillna(df[D.XB].median()).to_numpy(float)
    z = lambda v: (v - np.nanmean(v)) / np.nanstd(v)
    t = z(df.t_slope_correct_norm.fillna(df.t_slope_correct_norm.median()).to_numpy()) + \
        z(df.t_recent_shift_correct.fillna(df.t_recent_shift_correct.median()).to_numpy())
    X1 = np.c_[np.ones(len(X)), X]
    resid = t - X1 @ np.linalg.lstsq(X1, t, rcond=None)[0]
    lvl = df.b_correct_norm_win.fillna(df.b_correct_norm_win.median()).to_numpy()
    return z(resid), z(lvl)


def simulate_outcomes(units, scen, rng):
    """units: one row per unit with experiment_id, arm, arm_pos, s, T, L. Returns P(Y=1) per arm function."""
    exps = units.experiment_id.unique()
    delta = dict(zip(exps, rng.normal(0, 0.3, len(exps))))
    keys = units[["experiment_id", "arm", "arm_pos"]].drop_duplicates()
    theta = {(e, a): rng.normal(0, 0.02) for e, a in zip(keys.experiment_id, keys.arm)}
    if scen == "a_null":
        kap = {(e, a): 0.0 for e, a in zip(keys.experiment_id, keys.arm)}
    elif scen == "e_traj_mixed":
        kap = {(e, a): rng.normal(0, 0.03 / np.sqrt(2)) for e, a in zip(keys.experiment_id, keys.arm)}
    else:
        step = {"b_level": 0.03, "c_traj_small": 0.015, "d_traj_sesoi": 0.03, "f_traj_large": 0.06}[scen]
        kap = {(e, a): step * j for e, a, j in zip(keys.experiment_id, keys.arm, keys.arm_pos)}
    hcol = "L" if scen == "b_level" else "T"
    base = 1 / (1 + np.exp(-(units.s.to_numpy() + units.experiment_id.map(delta).to_numpy())))

    def prob(arm_series):
        k = [(e, a) for e, a in zip(units.experiment_id, arm_series)]
        th = np.array([theta[x] for x in k])
        ka = np.array([kap[x] for x in k])
        return np.clip(base + th + ka * units[hcol].to_numpy(), 0.01, 0.99)

    return prob


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=100)
    ap.add_argument("--jobs", type=int, default=12)
    ap.add_argument("--scenarios", default=",".join(SCEN))
    args = ap.parse_args()
    log = make_logger("04_simulation")
    df = D.load_frame("b24_k20")
    units = pd.read_parquet(D.DERIVED / "units_exp.parquet")[["unit_id", "experiment_id", "arm"]]
    units = units[units.unit_id.isin(df.unit_id)].reset_index(drop=True)
    units["arm_pos"] = units.groupby("experiment_id").arm.transform(lambda s: s.map({a: i for i, a in enumerate(sorted(s.unique()))}))
    per_unit = df.drop_duplicates("unit_id").set_index("unit_id")
    T, L = indices(per_unit.reset_index())
    units["T"] = pd.Series(T, index=per_unit.index).reindex(units.unit_id).to_numpy()
    units["L"] = pd.Series(L, index=per_unit.index).reindex(units.unit_id).to_numpy()
    units["s"] = per_unit.s.reindex(units.unit_id).to_numpy()
    log(f"simulation frame: {len(df):,} rows, {df.contrast_id.nunique()} contrasts, {len(units):,} units")
    tasks = [(s, r) for s in args.scenarios.split(",") for r in range(args.reps)]
    t0 = time.time()
    results = Parallel(n_jobs=args.jobs, verbose=10)(delayed(run_task)(df, units, s, r) for s, r in tasks)
    res = pd.DataFrame(results)
    res.to_csv(AUDIT / "simulation_reps.csv", index=False)
    log(f"done {len(tasks)} reps in {time.time() - t0:.0f}s")
    summarize(res)


def run_task(df, units, scen, rep):
    """One simulation replicate (runs inside a joblib worker)."""
    rng = np.random.default_rng([SEED, SCEN.index(scen), rep])
    prob = simulate_outcomes(units, scen, rng)
    y_unit = pd.Series(rng.binomial(1, prob(units.arm)), index=units.unit_id)
    d = df.copy()
    d["y"] = d.unit_id.map(y_unit).to_numpy()
    pos = pd.Index(units.unit_id).get_indexer(d.unit_id)
    sub = units.iloc[pos].reset_index(drop=True)
    prob_rows = simulate_outcomes_rows(units, scen, rep, sub)
    tau_true = prob_rows(d.arm1.to_numpy()) - prob_rows(d.arm0.to_numpy())
    out, _ = P.crossfit(d, D.XB, D.XT, seed=SEED + rep)
    res = {"scenario": scen, "rep": rep}
    for a, b, name in [("MB", "MBT", "primary"), ("MS", "MST", "s1"), ("M0", "MB", "s3")]:
        est, se = P.risk_diff(out, d, a, b)
        true = np.mean((tau_true - out[f"tau_{a}"]) ** 2 - (tau_true - out[f"tau_{b}"]) ** 2)
        res.update({f"{name}_est": est, f"{name}_se": se, f"{name}_true": true})
        est_c, se_c = P.risk_diff(out, d, a, b, weights="contrast")
        res.update({f"{name}_est_eqc": est_c, f"{name}_se_eqc": se_c})
    g, gse = P.policy_gain(out, d, "MB", "MBT")
    pv_true = np.mean(np.where(out.tau_MBT > 0, tau_true, 0) - np.where(out.tau_MB > 0, tau_true, 0))
    res.update({"s2_est": g, "s2_se": gse, "s2_true": pv_true})
    res["tau_sd_within_true"] = float(pd.Series(tau_true).groupby(d.contrast_id.values).std().mean())
    return res


def simulate_outcomes_rows(units, scen, rep, sub):
    """Re-create the same random parameters (same RNG stream) and evaluate arm probabilities for rows."""
    rng = np.random.default_rng([SEED, SCEN.index(scen), rep])
    exps = units.experiment_id.unique()
    delta = dict(zip(exps, rng.normal(0, 0.3, len(exps))))
    keys = units[["experiment_id", "arm", "arm_pos"]].drop_duplicates()
    theta = {(e, a): rng.normal(0, 0.02) for e, a in zip(keys.experiment_id, keys.arm)}
    if scen == "a_null":
        kap = {(e, a): 0.0 for e, a in zip(keys.experiment_id, keys.arm)}
    elif scen == "e_traj_mixed":
        kap = {(e, a): rng.normal(0, 0.03 / np.sqrt(2)) for e, a in zip(keys.experiment_id, keys.arm)}
    else:
        step = {"b_level": 0.03, "c_traj_small": 0.015, "d_traj_sesoi": 0.03, "f_traj_large": 0.06}[scen]
        kap = {(e, a): step * j for e, a, j in zip(keys.experiment_id, keys.arm, keys.arm_pos)}
    hcol = "L" if scen == "b_level" else "T"
    base = 1 / (1 + np.exp(-(sub.s.to_numpy() + sub.experiment_id.map(delta).to_numpy())))
    h = sub[hcol].to_numpy()
    e = sub.experiment_id.to_numpy()

    def prob(arms):
        th = np.array([theta[(ee, a)] for ee, a in zip(e, arms)])
        ka = np.array([kap[(ee, a)] for ee, a in zip(e, arms)])
        return np.clip(base + th + ka * h, 0.01, 0.99)

    return prob


def summarize(res):
    rows = []
    for scen, g in res.groupby("scenario"):
        n = len(g)
        row = {"scenario": scen, "reps": n, "tau_sd_within_true": g.tau_sd_within_true.mean()}
        for name in ("primary", "s1", "s3", "s2"):
            est, se, true = g[f"{name}_est"], g[f"{name}_se"], g[f"{name}_true"]
            lo, hi = est - 1.96 * se, est + 1.96 * se
            rej = (lo > 0).mean()
            row.update({
                f"{name}_mean_est": est.mean(), f"{name}_mean_true": true.mean(), f"{name}_bias": (est - true).mean(),
                f"{name}_emp_sd": est.std(), f"{name}_mean_se": se.mean(),
                f"{name}_coverage": ((lo <= true) & (true <= hi)).mean(),
                f"{name}_reject_lower_gt0": rej, f"{name}_reject_mcse": np.sqrt(rej * (1 - rej) / n),
                f"{name}_q05_est": est.quantile(0.05),
            })
        rows.append(row)
    summ = pd.DataFrame(rows)
    summ.to_csv(AUDIT / "simulation_summary.csv", index=False)
    save_json(summ.to_dict("records"), AUDIT / "simulation_summary.json")
    print(summ.T.to_string())


if __name__ == "__main__":
    main()
