"""Outcome-blind simulation for the effect-variation decomposition (step 2 of the follow-on study).

Real primary-sample structure (contrasts, arms, classes, teachers, districts, features, folds);
SYNTHETIC outcomes only. Per randomized unit i (experiment e, arm a, class k, teacher t, district d):
  P(Y=1) = clip( expit(s_i + delta_e) + theta_{e,a} + u^d_{e,a,d} + u^t_{e,a,t} + u^k_{e,a,k}, 0.01, 0.99 )
  delta_e ~ N(0, 0.3^2); theta_{e,a} ~ N(0, 0.025^2) (between-contrast SD ~3.5 pp, as observed);
  u^L ~ N(0, (sigma_L / sqrt 2)^2), so the CONTRAST-level component of level L has SD sigma_L.
Truth = the same pair estimator applied to the true row-level contrast effects (no sampling noise).

Outputs: data/derived/context_b24_k20.parquet (outcome-free context IDs),
         reports/audit_tables/decomp_sim_reps.csv, decomp_sim_summary.csv
Usage: python src/12_decomposition_simulation.py [--reps 100] [--jobs 14]
"""
import argparse
import time

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from common import AUDIT, DERIVED, SEED, connect, create_raw_views, make_logger
import data_access as D
import decomposition as Dc
import pipeline as P

SCEN = {"null": (0, 0, 0), "class_3pp": (0.03, 0, 0), "class_5pp": (0.05, 0, 0), "teacher_3pp": (0, 0.03, 0),
        "teacher_5pp": (0, 0.05, 0), "district_3pp": (0, 0, 0.03), "mixed_2pp": (0.02, 0.02, 0.02),
        "class_8pp": (0.08, 0, 0), "teacher_8pp": (0, 0.08, 0), "district_5pp": (0, 0, 0.05), "mixed_4pp": (0.04, 0.04, 0.04)}
LEVELS = ("class_id", "teacher", "district")


def components(edges_a, edges_b):
    """Connected components of a bipartite graph given as two aligned label arrays."""
    a = pd.factorize(edges_a)[0]
    b = pd.factorize(edges_b)[0] + a.max() + 1
    parent = np.arange(b.max() + 1)

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x
    for u, v in zip(a, b):
        ru, rv = find(u), find(v)
        if ru != rv:
            parent[ru] = rv
    return np.array([find(u) for u in a])


def build_frame(all_randomized=False):
    path = DERIVED / f"context_b24_k20{'_all' if all_randomized else ''}.parquet"
    df = D.load_frame("b24_k20", all_randomized=all_randomized)
    df["s"] = df.s.fillna(df.s.median())
    if not path.exists():
        con = connect(read_only=False)
        create_raw_views(con)
        pri = con.execute("""SELECT DISTINCT experiment_id, TRY_CAST(student_id AS BIGINT) student_id,
                             teacher_id, district_id FROM raw_cov_priors""").df()
        u = df.drop_duplicates("unit_id")[["unit_id", "experiment_id", "student_id", "class_id"]]
        u = u.merge(pri.drop_duplicates(["experiment_id", "student_id"]), on=["experiment_id", "student_id"],
                    how="left", validate="one_to_one")
        assert u.teacher_id.notna().all()
        # ~21% of units lack a district ID: each such teacher becomes its own pseudo-district (makes the
        # district component conservative; recorded in the feasibility note).
        u["district_id"] = u.district_id.astype("string").fillna("T" + u.teacher_id.astype(str))
        u["teacher"], u["district"] = Dc.nest_hierarchy(u.class_id.to_numpy(), u.teacher_id.to_numpy(), u.district_id.to_numpy())
        u["ctx_group"] = components(u.student_id.to_numpy(), u.district.to_numpy())
        u[["unit_id", "teacher", "district", "ctx_group"]].to_parquet(path, index=False)
    ctx = pd.read_parquet(path)
    return df.merge(ctx, on="unit_id", how="left", validate="many_to_one")


def draw_effects(units, arms_by_exp, col, sd, rng):
    combos = units[["experiment_id", col]].drop_duplicates().merge(arms_by_exp, on="experiment_id")
    combos["u"] = rng.normal(0, sd, len(combos)) if sd > 0 else 0.0
    return combos.set_index(["experiment_id", "arm", col]).u


def run_task(df, units, arms_by_exp, scen, rep):
    rng = np.random.default_rng([SEED, 12, list(SCEN).index(scen), rep])
    sig = SCEN[scen]
    exps = units.experiment_id.unique()
    delta = pd.Series(rng.normal(0, 0.3, len(exps)), index=exps)
    theta = arms_by_exp.assign(v=rng.normal(0, 0.025, len(arms_by_exp))).set_index(["experiment_id", "arm"]).v
    eff = {col: draw_effects(units, arms_by_exp, col, s / np.sqrt(2), rng) for col, s in zip(LEVELS, sig)}

    def prob(frame, arm):
        e = frame.experiment_id.to_numpy()
        p = 1 / (1 + np.exp(-(frame.s.to_numpy() + delta.reindex(e).to_numpy())))
        p = p + theta.reindex(pd.MultiIndex.from_arrays([e, arm])).to_numpy()
        for col in LEVELS:
            p = p + eff[col].reindex(pd.MultiIndex.from_arrays([e, arm, frame[col].to_numpy()])).to_numpy()
        return np.clip(p, 0.01, 0.99)

    y_unit = pd.Series(rng.binomial(1, prob(units, units.arm.to_numpy())), index=units.unit_id)
    d = df.copy()
    d["y"] = d.unit_id.map(y_unit).to_numpy()
    tau_true = prob(d, d.arm1.to_numpy()) - prob(d, d.arm0.to_numpy())
    out, _ = P.crossfit(d, D.XB, D.XT, seed=SEED + rep, models=("M0",), group_col="ctx_group")
    keys = (d.contrast_id.to_numpy(), d.district.to_numpy(), d.teacher.to_numpy(), d.class_id.to_numpy())
    est, se = Dc.decompose(*keys, out.phi.to_numpy())
    true, _ = Dc.decompose(*keys, tau_true, jackknife=False)
    res = {"scenario": scen, "rep": rep}
    for k in est:
        res.update({f"{k}_est": est[k], f"{k}_se": se[k], f"{k}_true": true[k]})
    return res


def summarize(res, tag=""):
    rows = []
    for scen, g in res.groupby("scenario", sort=False):
        for k in ("class", "teacher", "district", "total_context"):
            e, s, t = g[f"{k}_est"], g[f"{k}_se"], g[f"{k}_true"]
            null = res.loc[res.scenario == "null", f"{k}_est"]
            rows.append({"scenario": scen, "component": k, "reps": len(g), "true_mean": t.mean(),
                         "true_sd_pp": 100 * np.sqrt(max(t.mean(), 0)), "est_mean": e.mean(), "bias": (e - t).mean(),
                         "emp_sd": e.std(), "mean_jk_se": s.mean(),
                         "coverage": ((e - 1.96 * s <= t) & (t <= e + 1.96 * s)).mean(),
                         "power_lower_gt0": (e - 1.96 * s > 0).mean(),
                         "power_vs_null_q95": (e > null.quantile(0.95)).mean()})
    out = pd.DataFrame(rows)
    out.to_csv(AUDIT / f"decomp_sim_summary{tag}.csv", index=False)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=100)
    ap.add_argument("--jobs", type=int, default=14)
    ap.add_argument("--scenarios", default=",".join(SCEN))
    ap.add_argument("--population", choices=["primary", "all"], default="primary")
    args = ap.parse_args()
    tag = "" if args.population == "primary" else "_all"
    log = make_logger("12_decomposition_simulation")
    df = build_frame(args.population == "all")
    ue = pd.read_parquet(DERIVED / "units_exp.parquet")[["unit_id", "experiment_id", "arm"]]
    units = df.drop_duplicates("unit_id")[["unit_id", "s", "class_id", "teacher", "district"]].merge(ue, on="unit_id")
    arms_by_exp = ue[ue.experiment_id.isin(units.experiment_id)][["experiment_id", "arm"]].drop_duplicates().reset_index(drop=True)
    log(f"frame {len(df):,} rows; {units.teacher.nunique()} teachers; {units.district.nunique()} districts; "
        f"{df.ctx_group.nunique()} fold groups")
    tasks = [(s, r) for s in args.scenarios.split(",") for r in range(args.reps)]
    t0 = time.time()
    res = pd.DataFrame(Parallel(n_jobs=args.jobs, verbose=10)(delayed(run_task)(df, units, arms_by_exp, s, r) for s, r in tasks))
    res.to_csv(AUDIT / f"decomp_sim_reps{tag}.csv", index=False)
    log(f"done {len(tasks)} reps in {time.time() - t0:.0f}s")
    print(summarize(res, tag).round(5).to_string())


if __name__ == "__main__":
    main()
