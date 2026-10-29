"""Phase A (continued) — outcome-blind construction of units, pairwise contrasts, features and groups.

No experimental outcome is read or written by this script. Remnant outcomes are written
separately (remnant students were never randomized and do not overlap the experiment).

Outputs (data/derived, git-ignored):
  units_exp.parquet            one row per randomized student-experiment unit (assignment metadata)
  feat_exp_<variant>.parquet   features per unit for variants: b24_k20 (primary), b24_k10, b0_k20
  feat_rem_b24_k20.parquet     remnant features (+ remnant outcome) for the prognostic score
  contrasts.parquet            unit_id x contrast_id membership with z (0/1)
  groups.parquet               unit_id -> student-class connected-component group id
Audit tables (reports/audit_tables): contrast_summary.csv, sample_flow.csv
"""
import collections
import itertools

import numpy as np
import pandas as pd

from common import AUDIT, DERIVED, Timer, connect, create_raw_views, make_logger, save_json
import features as F

log = make_logger("02_build_sample")
MIN_PER_ARM = 56          # Sales et al. (2023) minimum per arm; fixed before any outcome analysis
VARIANTS = {"b24_k20": (86400, 20), "b24_k10": (86400, 10), "b0_k20": (60, 20)}


def build_units(con):
    """Randomized units from E-TRIALS assignment logs (labels are the source of truth)."""
    con.execute("CREATE OR REPLACE TABLE alogs AS SELECT * FROM raw_alogs")
    con.execute("CREATE OR REPLACE TABLE cond AS SELECT * FROM raw_exp_conditions2")
    con.execute("CREATE OR REPLACE TABLE tgt AS SELECT * FROM raw_exp_targets")
    flow = []
    flow.append(("E-TRIALS assignment-log rows", int(con.execute("SELECT count(*) FROM alogs").fetchone()[0])))
    con.execute("""CREATE OR REPLACE TABLE units_all AS
        SELECT a.assignment_log_id AS unit_id, a.experiment_id, a.assigned_condition AS arm,
               t.student_id, t.class_id, t.assignment_start_time, t.has_due_date,
               c.in_control, c.in_treatment,
               count(*) OVER (PARTITION BY a.assignment_log_id) AS n_alog_rows
        FROM alogs a JOIN tgt t USING (assignment_log_id) LEFT JOIN cond c USING (assignment_log_id)""")
    flow.append(("... linked to a target row", int(con.execute("SELECT count(*) FROM units_all").fetchone()[0])))
    con.execute("DELETE FROM units_all WHERE arm = 'Not Assigned'")
    flow.append(("... excluding 'Not Assigned' (never randomized)", int(con.execute("SELECT count(*) FROM units_all").fetchone()[0])))
    con.execute("DELETE FROM units_all WHERE n_alog_rows > 1")
    flow.append(("... excluding assignment logs listed under >1 condition (conflicting)",
                 int(con.execute("SELECT count(*) FROM units_all").fetchone()[0])))
    # Released binary flags must agree with the label type where present (Control* -> (1,0); Treatment* -> (0,1)).
    con.execute("""DELETE FROM units_all WHERE in_control IS NOT NULL AND NOT (
                     (arm LIKE 'Control%' AND in_control = 1 AND in_treatment = 0) OR
                     (arm LIKE 'Treatment%' AND in_control = 0 AND in_treatment = 1))""")
    flow.append(("... excluding rows whose released flags contradict the label",
                 int(con.execute("SELECT count(*) FROM units_all").fetchone()[0])))
    dup = con.execute("SELECT count(*) FROM (SELECT student_id, experiment_id FROM units_all GROUP BY 1,2 HAVING count(*)>1)").fetchone()[0]
    if dup:
        raise AssertionError(f"{dup} student-experiment pairs have >1 randomized unit")
    units = con.execute("SELECT unit_id, experiment_id, arm, student_id, class_id, assignment_start_time, has_due_date "
                        "FROM units_all").df()
    return units, flow


def build_features(con, units_view, hist_view, act_view, buffer_s, ks):
    with Timer(f"prior rows buffer={buffer_s}", log):
        con.execute(f"CREATE OR REPLACE TEMP TABLE prior AS {F.prior_history_sql(units_view, hist_view, buffer_s)}")
    life = con.execute(F.lifetime_sql("prior")).df()
    acts = con.execute(F.actions_sql(units_view, act_view, buffer_s)).df()
    units = con.execute(f"SELECT unit_id, assignment_start_time, has_due_date FROM {units_view}").df()
    out = {}
    for K in ks:
        with Timer(f"window K={K}", log):
            win = con.execute(f"""SELECT * FROM (
                    SELECT *, row_number() OVER (PARTITION BY unit_id ORDER BY h_start DESC) AS rn_stats
                    FROM prior WHERE has_stats) WHERE rn_stats <= {K}""").df()
            wf = F.window_features(win, K)
        out[K] = F.assemble(units, life, acts, wf)
    return out


def components(units):
    """Connected components of the bipartite student-class graph (all randomized units)."""
    parent = {}

    def find(x):
        while parent.setdefault(x, x) != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for s, c in units[["student_id", "class_id"]].drop_duplicates().itertuples(index=False):
        a, b = find(("s", s)), find(("c", c))
        if a != b:
            parent[a] = b
    roots = {s: find(("s", s)) for s in units.student_id.unique()}
    ids = {r: i for i, r in enumerate(sorted(set(roots.values()), key=str))}
    return units.student_id.map(lambda s: ids[roots[s]]).astype(int)


def build_contrasts(units):
    rows = []
    for exp, g in units.groupby("experiment_id"):
        arms = sorted(g.arm.unique())
        for a0, a1 in itertools.combinations(arms, 2):
            cid = f"{exp}|{a0}|{a1}"
            sub = g[g.arm.isin([a0, a1])]
            rows.append(pd.DataFrame({"unit_id": sub.unit_id.values, "contrast_id": cid, "experiment_id": exp,
                                      "arm0": a0, "arm1": a1, "z": (sub.arm == a1).astype(int).values}))
    return pd.concat(rows, ignore_index=True)


def main():
    con = connect()
    create_raw_views(con)
    S = {}
    units, flow = build_units(con)
    S["sample_flow_units"] = flow
    units["group"] = components(units)
    con.register("units_df", units)
    con.execute("CREATE OR REPLACE TABLE units_exp AS SELECT * FROM units_df")
    units.to_parquet(DERIVED / "units_exp.parquet", index=False)
    units[["unit_id", "group", "student_id", "class_id"]].to_parquet(DERIVED / "groups.parquet", index=False)
    sizes = units.groupby("group").student_id.nunique()
    S["groups"] = {"n_groups": int(len(sizes)), "largest_students": int(sizes.max()),
                   "largest_share": float(sizes.max() / units.student_id.nunique())}

    # Experimental features (three pre-specified variants).
    feats = {}
    for buf in sorted({b for b, _ in VARIANTS.values()}):
        ks = sorted({k for b, k in VARIANTS.values() if b == buf})
        res = build_features(con, "units_exp", "raw_exp_assign", "raw_exp_actions", buf, ks)
        for name, (b, k) in VARIANTS.items():
            if b == buf:
                feats[name] = res[k]
                res[k].to_parquet(DERIVED / f"feat_exp_{name}.parquet", index=False)

    # Remnant features + remnant outcome (for the prognostic score only).
    con.execute("""CREATE OR REPLACE TABLE units_rem AS
                   SELECT assignment_log_id AS unit_id, student_id, class_id, assignment_start_time, has_due_date,
                          assignment_completed FROM raw_rem_targets""")
    rem = build_features(con, "units_rem", "raw_rem_assign", "raw_rem_actions", 86400, [20])[20]
    rem = rem.merge(con.execute("SELECT unit_id, class_id, student_id, assignment_completed FROM units_rem").df(), on="unit_id")
    rem.to_parquet(DERIVED / "feat_rem_b24_k20.parquet", index=False)
    S["remnant_units"] = int(len(rem))
    S["remnant_units_with_10_stats"] = int((rem.n_prior_stats >= 10).sum())

    # Pairwise contrasts and outcome-blind eligibility.
    con_df = build_contrasts(units)
    con_df.to_parquet(DERIVED / "contrasts.parquet", index=False)
    f20, f10 = feats["b24_k20"], feats["b24_k10"]
    m = con_df.merge(f20[["unit_id", "n_prior_stats"]], on="unit_id").merge(
        feats["b0_k20"][["unit_id", "n_prior_stats"]].rename(columns={"n_prior_stats": "n_prior_stats_b0"}), on="unit_id")
    m["h10"] = m.n_prior_stats >= 10
    m["h5"] = m.n_prior_stats >= 5
    m["h10_b0"] = m.n_prior_stats_b0 >= 10
    summ = m.groupby(["contrast_id", "experiment_id", "arm0", "arm1"]).agg(
        n0_all=("z", lambda s: int((s == 0).sum())), n1_all=("z", lambda s: int((s == 1).sum())),
        n0_h10=("z", lambda s: int(((s == 0) & m.loc[s.index, "h10"]).sum())),
        n1_h10=("z", lambda s: int(((s == 1) & m.loc[s.index, "h10"]).sum())),
        n0_h5=("z", lambda s: int(((s == 0) & m.loc[s.index, "h5"]).sum())),
        n1_h5=("z", lambda s: int(((s == 1) & m.loc[s.index, "h5"]).sum())),
        n0_h10_b0=("z", lambda s: int(((s == 0) & m.loc[s.index, "h10_b0"]).sum())),
        n1_h10_b0=("z", lambda s: int(((s == 1) & m.loc[s.index, "h10_b0"]).sum())),
    ).reset_index()
    for tag in ("all", "h10", "h5", "h10_b0"):
        summ[f"eligible_{tag}"] = (summ[f"n0_{tag}"] >= MIN_PER_ARM) & (summ[f"n1_{tag}"] >= MIN_PER_ARM)
    summ["share1_all"] = summ.n1_all / (summ.n0_all + summ.n1_all)
    summ.to_csv(AUDIT / "contrast_summary.csv", index=False)
    for tag in ("all", "h10", "h5", "h10_b0"):
        e = summ[summ[f"eligible_{tag}"]]
        S[f"contrasts_eligible_{tag}"] = {
            "contrasts": int(len(e)), "experiments": int(e.experiment_id.nunique()),
            "unit_contrast_rows": int(e[f"n0_{tag}"].sum() + e[f"n1_{tag}"].sum()),
        }
    S["contrasts_total"] = int(len(summ))
    S["experiments_with_contrasts"] = int(summ.experiment_id.nunique())
    prim = summ[summ.eligible_h10]
    pm = m[m.contrast_id.isin(prim.contrast_id) & m.h10]
    S["primary_distinct_units"] = int(pm.unit_id.nunique())
    S["primary_distinct_students"] = int(units[units.unit_id.isin(pm.unit_id)].student_id.nunique())
    S["primary_contrast_size_quantiles"] = (prim.n0_h10 + prim.n1_h10).quantile([0, .25, .5, .75, 1]).to_dict()
    S["primary_contrasts_per_experiment"] = prim.groupby("experiment_id").size().value_counts().sort_index().to_dict()
    # Feature missingness among primary units (no outcomes involved).
    fp = f20[f20.unit_id.isin(pm.unit_id)]
    S["primary_feature_missing_share"] = {c: float(fp[c].isna().mean()) for c in F.LEVEL_FEATURES + F.TRAJECTORY_FEATURES}
    pd.DataFrame(flow, columns=["step", "n"]).to_csv(AUDIT / "sample_flow.csv", index=False)
    save_json(S, AUDIT / "sample_summary.json")
    log("sample build complete")


if __name__ == "__main__":
    main()
