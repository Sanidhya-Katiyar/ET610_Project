"""POST-UNBLINDING EXPLORATORY: consequences of the data issues found in the audit.

1. Merged arms: contrasts that the released binary flags cannot see.
2. Leakage: naive feature construction (history not cut before the target) - prognostic inflation,
   arm imbalance (post-treatment information) and spurious heterogeneity (naive Delta_R).
3. Released precomputed priors: prognostic value and arm balance.
4. Undocumented interventions: power of a descriptor meta-regression across contrasts.
5. Follow-up truncation: summary of the earlier feasibility audit.

Output: reports/results_tables/data_gaps.json, data_gaps_leakage.csv, data_gaps_flag_contrasts.csv
"""
import importlib
import json

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import roc_auc_score

from common import AUDIT, DERIVED, REPORTS, SEED, connect, create_raw_views, make_logger, save_json
import data_access as D
import pipeline as P

A = importlib.import_module("05_primary_analysis")
B02 = importlib.import_module("02_build_sample")
log = make_logger("14_data_gap_analyses")
OUT = REPORTS / "results_tables"
CHECK = ["b_completion_rate_all", "b_correct_all", "b_correct_norm_win", "t_completion_shift", "t_recent_shift_correct",
         "t_slope_correct_norm"]
RELEASED = ["student_prior_assignments_percent_completed", "student_prior_average_problem_correctness",
            "student_prior_assignments_started"]


def balance_q0(contrast, z, x):
    d = pd.DataFrame({"c": contrast, "z": z, "x": x}).dropna()
    g = d.groupby(["c", "z"]).x.agg(["mean", "var", "count"]).unstack("z")
    g = g[(g["count"] >= 2).all(axis=1)]
    diff = g[("mean", 1)] - g[("mean", 0)]
    se = np.sqrt(g[("var", 1)] / g[("count", 1)] + g[("var", 0)] / g[("count", 0)])
    ok = se > 0
    zc = (diff[ok] / se[ok]).to_numpy()
    q = float(np.sum(zc ** 2))
    return {"Q0": q, "df": int(len(zc)), "p": float(stats.chi2.sf(q, len(zc))), "share_abs_z_gt_1.96": float(np.mean(np.abs(zc) > 1.96))}


def auc(y, x):
    ok = ~np.isnan(x)
    a = roc_auc_score(y[ok], x[ok])
    return float(max(a, 1 - a))


def winsorize(feats):
    bounds = json.loads((AUDIT / "winsor_bounds.json").read_text())
    for c, (lo, hi) in bounds.items():
        if c in feats:
            feats[c] = feats[c].clip(lo, hi)
    return feats


def flags_part(con, outcomes):
    units = pd.read_parquet(DERIVED / "units_exp.parquet")[["unit_id", "experiment_id", "arm"]]
    fl = con.execute("SELECT assignment_log_id AS unit_id, in_control, in_treatment FROM raw_exp_conditions2").df()
    u = units.merge(fl, on="unit_id", how="left").merge(outcomes, on="unit_id", how="left")
    u["flag"] = np.select([(u.in_control == 1) & (u.in_treatment == 0), (u.in_control == 0) & (u.in_treatment == 1)],
                          ["control", "treatment"], default="none")
    arm_flag = u.groupby(["experiment_id", "arm"]).flag.agg(lambda s: s.mode().iloc[0])
    con_df = pd.read_parquet(DERIVED / "contrasts.parquet")[["contrast_id", "experiment_id", "arm0", "arm1"]].drop_duplicates()
    con_df["f0"] = arm_flag.reindex(pd.MultiIndex.from_frame(con_df[["experiment_id", "arm0"]])).to_numpy()
    con_df["f1"] = arm_flag.reindex(pd.MultiIndex.from_frame(con_df[["experiment_id", "arm1"]])).to_numpy()
    con_df["visible_with_flags"] = (con_df.f0 != con_df.f1) & (con_df.f0 != "none") & (con_df.f1 != "none")
    rate = u.groupby(["experiment_id", "arm"]).assignment_completed.agg(["mean", "count"])
    m0 = rate.reindex(pd.MultiIndex.from_frame(con_df[["experiment_id", "arm0"]]))
    m1 = rate.reindex(pd.MultiIndex.from_frame(con_df[["experiment_id", "arm1"]]))
    con_df["ate"] = m1["mean"].to_numpy() - m0["mean"].to_numpy()
    con_df["se"] = np.sqrt(m1["mean"].to_numpy() * (1 - m1["mean"].to_numpy()) / m1["count"].to_numpy()
                           + m0["mean"].to_numpy() * (1 - m0["mean"].to_numpy()) / m0["count"].to_numpy())
    con_df["significant"] = np.abs(con_df.ate / con_df.se) > 1.96
    con_df.drop(columns=["arm0", "arm1"]).to_csv(OUT / "data_gaps_flag_contrasts.csv", index=False)
    inv = con_df[~con_df.visible_with_flags]
    n_arms = units.groupby("experiment_id").arm.nunique()
    return {"randomized_units": int(len(u)), "share_units_flag_none": float((u.flag == "none").mean()),
            "experiments": int(len(n_arms)), "experiments_ge3_arms": int((n_arms >= 3).sum()),
            "contrasts": int(len(con_df)), "contrasts_invisible_with_flags": int(len(inv)),
            "invisible_significant": int(inv.significant.sum()),
            "invisible_expected_significant_under_null": float(0.05 * len(inv)),
            "invisible_mean_abs_ate_pp": float(100 * inv.ate.abs().mean()),
            "visible_mean_abs_ate_pp": float(100 * con_df[con_df.visible_with_flags].ate.abs().mean())}


def leakage_part(con, outcomes):
    con.execute(f"CREATE OR REPLACE TEMP VIEW units_x AS SELECT * FROM read_parquet('{(DERIVED / 'units_exp.parquet').as_posix()}')")
    variants = {"leak_self": 0.0, "leak_all": -1e10}
    feats = {}
    for name, buf in variants.items():
        f = B02.build_features(con, "units_x", "raw_exp_assign", "raw_exp_actions", buf, [20])[20]
        feats[name] = winsorize(f)
        log(f"built {name}")
    df = A.join_outcomes(D.load_frame("b24_k20"), outcomes)
    rel = con.execute(f"SELECT assignment_log_id AS unit_id, {', '.join(RELEASED)} FROM raw_exp_targets").df()
    unit_y = df.drop_duplicates("unit_id")[["unit_id", "y"]]
    rows = []
    sources = {"clean": df.drop_duplicates("unit_id")[["unit_id"] + D.XB + D.XT]}
    sources.update({k: v[["unit_id"] + D.XB + D.XT] for k, v in feats.items()})
    for src, f in sources.items():
        uy = unit_y.merge(f, on="unit_id", how="left")
        rows_df = df[["contrast_id", "z", "unit_id"]].merge(f, on="unit_id", how="left")
        for col in CHECK:
            rows.append({"source": src, "feature": col, "auc_completion": auc(uy.y.to_numpy(), uy[col].to_numpy(float)),
                         **balance_q0(rows_df.contrast_id, rows_df.z, rows_df[col].to_numpy(float))})
    uy = unit_y.merge(rel, on="unit_id", how="left")
    rows_df = df[["contrast_id", "z", "unit_id"]].merge(rel, on="unit_id", how="left")
    for col in RELEASED:
        rows.append({"source": "released_precomputed", "feature": col, "auc_completion": auc(uy.y.to_numpy(), uy[col].to_numpy(float)),
                     **balance_q0(rows_df.contrast_id, rows_df.z, rows_df[col].to_numpy(float))})
    tab = pd.DataFrame(rows)
    tab.to_csv(OUT / "data_gaps_leakage.csv", index=False)
    # Naive Delta_R: leaky trajectory features (clean levels), and fully leaky features.
    naive = {}
    for name, f in feats.items():
        d_t = df.drop(columns=D.XT).merge(f[["unit_id"] + D.XT], on="unit_id", how="left")
        out, _ = P.crossfit(d_t, D.XB, D.XT, seed=SEED, models=("M0", "MB", "MBT"))
        e, s = P.risk_diff(out, d_t, "MB", "MBT")
        d_all = df.drop(columns=D.XB + D.XT).merge(f[["unit_id"] + D.XB + D.XT], on="unit_id", how="left")
        out2, _ = P.crossfit(d_all, D.XB, D.XT, seed=SEED, models=("M0", "MB", "MBT"))
        e2, s2 = P.risk_diff(out2, d_all, "M0", "MBT")
        naive[name] = {"dR_MB_to_MBT_leaky_traj": e, "se": s, "dR_M0_to_MBT_all_leaky": e2, "se_all": s2}
        log(f"naive dR {name}: {naive[name]}")
    return tab, naive


def descriptor_power():
    itt = pd.read_csv(OUT / "contrast_itt.csv")
    tau = json.loads((OUT / "results.json").read_text())["itt"]["all_randomized"]["tau_sd_across_contrasts"]
    W = np.sum(1 / (itt.se_all ** 2 + tau ** 2))
    res = {"tau_between_pp": 100 * tau, "contrasts": int(len(itt))}
    for pi in (0.5, 0.2):
        mde = 2.8 * np.sqrt(1 / (pi * W) + 1 / ((1 - pi) * W))
        res[f"mde_pp_split_{int(pi * 100)}"] = float(100 * mde)
        res[f"share_between_var_explained_at_mde_split_{int(pi * 100)}"] = float(mde ** 2 * pi * (1 - pi) / tau ** 2)
    return res


def main():
    con = connect(read_only=False)
    create_raw_views(con)
    A.check_protocol()
    outcomes = A.load_outcomes()[["unit_id", "assignment_completed"]]
    res = {"note": "post-unblinding exploratory consequence analyses"}
    res["flags"] = flags_part(con, outcomes)
    log(f"flags: {res['flags']}")
    res["descriptor_meta_regression_power"] = descriptor_power()
    fe = json.loads((AUDIT / "followup_feasibility.json").read_text())
    sc = json.loads((OUT / "followup_selection_check.json").read_text())
    res["followup"] = {"students_with_history_after_last_target": fe["history_extraction"]["students_with_rows_after_last_target"],
                       "share_units_no_window": fe["share_units_no_window"], "S7_p": sc["S7"]["p_Q0"], "S7_tau": sc["S7"]["dl_tau"]}
    tab, naive = leakage_part(con, outcomes)
    res["naive_delta_r"] = naive
    save_json(res, OUT / "data_gaps.json")
    print(json.dumps(res, indent=1, default=str))
    print(tab.round(4).to_string())


if __name__ == "__main__":
    main()
