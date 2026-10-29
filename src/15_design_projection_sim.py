"""Design projection for the context decomposition: how does precision scale with students per cluster?

Synthetic outcomes only. Each randomized unit is replicated m times inside its own class (same class,
teacher, district and features); replicate r >= 1 gets arms by a within-experiment permutation of the
real arms (preserving allocation shares) and fresh outcomes. m = 1 is the existing all-randomized
simulation (reports/audit_tables/decomp_sim_reps_all.csv).

Output: reports/audit_tables/design_projection_reps.csv, design_projection.csv
Usage: python src/15_design_projection_sim.py [--reps 30] [--jobs 3] [--m 2,3]
"""
import argparse
import importlib
import time

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy import stats

from common import AUDIT, DERIVED, SEED, make_logger

S = importlib.import_module("12_decomposition_simulation")
SCENARIOS = ["null", "mixed_2pp"]
OFFSET = 10 ** 10


def expand(df, units, m, seed):
    """Replicate every unit m times within its class; returns (rows frame, unit frame)."""
    if m == 1:
        return df, units
    rng = np.random.default_rng([seed, m])
    contrasts = df[["contrast_id", "experiment_id", "arm0", "arm1"]].drop_duplicates()
    unit_cols = [c for c in df.columns if c not in ("contrast_id", "arm0", "arm1", "z")]
    per_unit = df.drop_duplicates("unit_id")[unit_cols]
    rows, ulist = [df], [units]
    for r in range(1, m):
        u = units.copy()
        u["arm"] = u.groupby("experiment_id").arm.transform(lambda a: rng.permutation(a.to_numpy()))
        u["unit_id"] = u.unit_id + r * OFFSET
        ulist.append(u)
        pu = per_unit.copy()
        pu["unit_id"] = pu.unit_id + r * OFFSET
        pu["student_id"] = pu.student_id + r * OFFSET
        pu = pu.merge(u[["unit_id", "arm"]], on="unit_id")
        x = pu.merge(contrasts, on="experiment_id")
        x = x[(x.arm == x.arm0) | (x.arm == x.arm1)]
        x["z"] = (x.arm == x.arm1).astype(int)
        rows.append(x[df.columns])
    return pd.concat(rows, ignore_index=True), pd.concat(ulist, ignore_index=True)


def task(df, units, arms_by_exp, m, scen, rep):
    d, u = expand(df, units, m, SEED + rep)
    res = S.run_task(d, u, arms_by_exp, scen, rep)
    res["m"] = m
    res["rows"] = len(d)
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=30)
    ap.add_argument("--jobs", type=int, default=3)
    ap.add_argument("--m", default="2,3")
    args = ap.parse_args()
    log = make_logger("15_design_projection_sim")
    df = S.build_frame(all_randomized=True)
    ue = pd.read_parquet(DERIVED / "units_exp.parquet")[["unit_id", "experiment_id", "arm"]]
    units = df.drop_duplicates("unit_id")[["unit_id", "s", "class_id", "teacher", "district"]].merge(ue, on="unit_id")
    arms_by_exp = ue[ue.experiment_id.isin(units.experiment_id)][["experiment_id", "arm"]].drop_duplicates().reset_index(drop=True)
    ms = [int(x) for x in args.m.split(",")]
    tasks = [(m, s, r) for m in ms for s in SCENARIOS for r in range(args.reps)]
    t0 = time.time()
    res = pd.DataFrame(Parallel(n_jobs=args.jobs, verbose=10)(delayed(task)(df, units, arms_by_exp, m, s, r) for m, s, r in tasks))
    base = pd.read_csv(AUDIT / "decomp_sim_reps_all.csv", keep_default_na=False)
    base = base[base.scenario.isin(SCENARIOS)].assign(m=1, rows=len(df))
    allres = pd.concat([base, res], ignore_index=True)
    allres.to_csv(AUDIT / "design_projection_reps.csv", index=False)
    log(f"done {len(tasks)} tasks in {time.time() - t0:.0f}s")
    out = []
    for m, g in allres.groupby("m"):
        nul, alt = g[g.scenario == "null"], g[g.scenario == "mixed_2pp"]
        se = alt.total_context_se.mean()
        true = alt.total_context_true.mean()
        out.append({"m": m, "rows": int(g.rows.iloc[0]), "reps_alt": len(alt), "se_total": se,
                    "emp_sd_null": nul.total_context_est.std(), "true_V_alt": true,
                    "true_sd_pp_alt": 100 * np.sqrt(true),
                    "power_sim_alt": float((alt.total_context_est - 1.96 * alt.total_context_se > 0).mean()),
                    "power_analytic_alt": float(stats.norm.cdf(true / se - 1.96)),
                    "upper_sd_pp_if_zero": 100 * np.sqrt(1.96 * se)})
    out = pd.DataFrame(out)
    out.to_csv(AUDIT / "design_projection.csv", index=False)
    print(out.round(5).to_string())


if __name__ == "__main__":
    main()
