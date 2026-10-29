"""POST-UNBLINDING EXPLORATORY: randomization diagnostics and robustness of the headline results.

Diagnostics (pre-treatment only): arm clustering within classes (dispersion test with permutation
reference) and arm imbalance in calendar time, per contrast.
Robustness: on the "clean-randomization" subset (class-clustering p >= 0.05 and |z_time| <= 3), re-estimate
  - between-contrast heterogeneity (difference in means and covariate-adjusted DR ATEs),
  - the total-context decomposition.
Output: reports/results_tables/randomization_robustness.json, randomization_by_contrast.csv
"""
import importlib

import numpy as np
import pandas as pd
from scipy import stats

from common import DERIVED, REPORTS, SEED, make_logger, save_json
import data_access as D
import decomposition as Dc
import pipeline as P

A = importlib.import_module("05_primary_analysis")
S = importlib.import_module("12_decomposition_simulation")
log = make_logger("16_randomization_robustness")
OUT = REPORTS / "results_tables"


def class_dispersion(df):
    rows = []
    for c, g in df.groupby("contrast_id"):
        p = g.z.mean()
        cl = g.groupby("class_id").z.agg(["sum", "count"])
        cl = cl[cl["count"] >= 2]
        if len(cl) < 2 or p in (0, 1):
            rows.append((c, np.nan, 0))
            continue
        rows.append((c, float(((cl["sum"] - cl["count"] * p) ** 2 / (cl["count"] * p * (1 - p))).sum()), len(cl) - 1))
    return pd.DataFrame(rows, columns=["contrast_id", "chi2", "df"]).set_index("contrast_id")


def time_z(df):
    g = df.groupby(["contrast_id", "z"]).start.agg(["mean", "var", "count"]).unstack("z")
    return (g[("mean", 1)] - g[("mean", 0)]) / np.sqrt(g[("var", 1)] / g[("count", 1)] + g[("var", 0)] / g[("count", 0)])


def contrast_ates(df, phi):
    d = df.assign(phi=phi)
    g = d.groupby(["contrast_id", "z"]).y.agg(["mean", "count"]).unstack("z")
    dm = g[("mean", 1)] - g[("mean", 0)]
    dm_se = np.sqrt(g[("mean", 1)] * (1 - g[("mean", 1)]) / g[("count", 1)] + g[("mean", 0)] * (1 - g[("mean", 0)]) / g[("count", 0)])
    dr = d.groupby("contrast_id").phi.agg(["mean", "std", "count"])
    return pd.DataFrame({"dm": dm, "dm_se": dm_se, "dr": dr["mean"], "dr_se": dr["std"] / np.sqrt(dr["count"])})


def main():
    A.check_protocol()
    df = A.join_outcomes(S.build_frame(all_randomized=True), A.load_outcomes())
    st = pd.read_parquet(DERIVED / "units_exp.parquet")[["unit_id", "assignment_start_time"]].rename(columns={"assignment_start_time": "start"})
    df = df.merge(st, on="unit_id", how="left")
    disp = class_dispersion(df)
    disp["p_class"] = stats.chi2.sf(disp.chi2, disp.df)
    disp["z_time"] = time_z(df)
    rng = np.random.default_rng(SEED)
    perm_ratios = []
    for _ in range(5):
        dp = df.assign(z=df.groupby("contrast_id").z.transform(lambda s: rng.permutation(s.to_numpy())))
        dd = class_dispersion(dp)
        perm_ratios.append(float(dd.chi2.sum() / dd.df.sum()))
    clean = disp.index[(disp.p_class.fillna(1) >= 0.05) & (disp.z_time.abs() <= 3)]
    disp["clean"] = disp.index.isin(clean)
    disp.to_csv(OUT / "randomization_by_contrast.csv")
    res = {"class_dispersion_ratio": float(disp.chi2.sum() / disp.df.sum()), "permutation_ratios": perm_ratios,
           "share_contrasts_p_class_lt_05": float((disp.p_class < 0.05).mean()),
           "contrasts_abs_z_time_gt3": int((disp.z_time.abs() > 3).sum()),
           "clean_contrasts": int(len(clean)), "contrasts": int(len(disp))}
    out, _ = P.crossfit(df, D.XB, D.XT, seed=SEED, models=("M0",), group_col="ctx_group")
    ates = contrast_ates(df, out.phi.to_numpy())
    for label, idx in (("all", ates.index), ("clean", ates.index.intersection(clean))):
        a = ates.loc[idx]
        res[f"between_{label}"] = {"diff_in_means": A.dersimonian_laird(a.dm.to_numpy(), a.dm_se.to_numpy()),
                                   "dr_adjusted": A.dersimonian_laird(a.dr.to_numpy(), a.dr_se.to_numpy())}
    dc = df[df.contrast_id.isin(clean)].reset_index(drop=True)
    oc = out[df.contrast_id.isin(clean).to_numpy()].reset_index(drop=True)
    est, se = Dc.decompose(dc.contrast_id.to_numpy(), dc.district.to_numpy(), dc.teacher.to_numpy(), dc.class_id.to_numpy(), oc.phi.to_numpy())
    res["decomposition_clean"] = {k: {"V": est[k], "se": se[k], "sd_pp": 100 * float(np.sqrt(max(est[k], 0))),
                                      "sd_upper_pp": 100 * float(np.sqrt(max(est[k] + 1.96 * se[k], 0)))} for k in est}
    res["rows_clean"] = int(len(dc))
    save_json(res, OUT / "randomization_robustness.json")
    log(f"randomization robustness: {res}")
    for k, v in res.items():
        print(k, v)


if __name__ == "__main__":
    main()
