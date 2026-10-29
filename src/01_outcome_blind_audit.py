"""Phase A — outcome-blind schema and design audit.

RULE: this script never tabulates, plots or models an outcome by treatment arm.
It inspects schema, IDs, flags, timestamps, arm sizes, history availability and
*overall* outcome availability only. Outputs: reports/audit_tables/*.csv and
reports/audit_tables/audit_summary.json.

Usage: python src/01_outcome_blind_audit.py
"""
import collections

import numpy as np
import pandas as pd

from common import AUDIT, Timer, connect, create_raw_views, make_logger, save_json

log = make_logger("01_outcome_blind_audit")
S = {}  # summary dict


def q(con, sql):
    return con.execute(sql).df()


def one(con, sql):
    return con.execute(sql).fetchone()


def main():
    con = connect()
    create_raw_views(con)

    # ---- 1. Parseability, row counts, column null counts -------------------------------
    files = {}
    for v in ["exp_conditions", "exp_targets", "exp_assign", "exp_actions", "rem_targets", "rem_assign",
              "rem_actions", "exp_norm_map"]:
        with Timer(f"profile {v}", log):
            cols = q(con, f"DESCRIBE raw_{v}")
            exprs = ", ".join([f'count("{c}") AS "{c}"' for c in cols.column_name])
            counts = q(con, f"SELECT count(*) AS __n, {exprs} FROM raw_{v}").iloc[0]
            n = int(counts["__n"])
            prof = pd.DataFrame({
                "column": cols.column_name, "duckdb_type": cols.column_type,
                "non_null": [int(counts[c]) for c in cols.column_name],
            })
            prof["null_frac"] = 1 - prof.non_null / n
            prof.to_csv(AUDIT / f"schema_{v}.csv", index=False)
            files[v] = {"rows": n, "columns": len(cols)}
            log(f"{v}: {n:,} rows x {len(cols)} cols")
    S["files"] = files

    # Materialise the two small experimental tables for fast joins.
    con.execute("CREATE OR REPLACE TABLE cond AS SELECT * FROM raw_exp_conditions2")
    con.execute("CREATE OR REPLACE TABLE tgt AS SELECT * FROM raw_exp_targets")
    con.execute("CREATE OR REPLACE TABLE rtgt AS SELECT * FROM raw_rem_targets")

    # ---- 2. Assignment flags ------------------------------------------------------------
    flags = q(con, "SELECT in_control, in_treatment, count(*) n FROM cond GROUP BY ALL ORDER BY ALL")
    flags.to_csv(AUDIT / "flag_combinations.csv", index=False)
    S["flag_combinations"] = flags.to_dict("records")
    S["cond_rows"], S["cond_distinct_alog"], S["cond_distinct_user"] = map(int, one(
        con, "SELECT count(*), count(DISTINCT assignment_log_id), count(DISTINCT user_id) FROM cond"))
    S["cond_dup_alog"] = int(one(con, "SELECT count(*) FROM (SELECT assignment_log_id FROM cond "
                                       "GROUP BY 1 HAVING count(*)>1)")[0])

    # ---- 2b. Condition labels from the per-experiment E-TRIALS files ----------------------
    con.execute("CREATE OR REPLACE TABLE alogs AS SELECT * FROM raw_alogs")
    con.execute("CREATE OR REPLACE TABLE covp AS SELECT DISTINCT * FROM raw_cov_priors")
    r = one(con, "SELECT count(*), count(DISTINCT experiment_id), count(DISTINCT assignment_log_id), "
                 "count(DISTINCT student_id) FROM alogs")
    S["alogs"] = dict(zip(["rows", "experiments", "distinct_alog", "distinct_students"], map(int, r)))
    S["alogs_dup_alog"] = int(one(con, "SELECT count(*) FROM (SELECT assignment_log_id FROM alogs "
                                       "GROUP BY 1 HAVING count(*)>1)")[0])
    S["alogs_raw_label_variants"] = q(con, "SELECT assigned_condition_raw, assigned_condition, count(*) n "
                                           "FROM alogs GROUP BY 1,2 ORDER BY 3 DESC").to_dict("records")
    # Do whitespace variants co-occur in one experiment (would indicate distinct arms)?
    S["experiments_with_whitespace_label_collisions"] = int(one(con, """
        SELECT count(*) FROM (SELECT experiment_id, assigned_condition FROM alogs
                              GROUP BY 1,2 HAVING count(DISTINCT assigned_condition_raw)>1)""")[0])
    # Flags versus labels (cross-tab of assignment metadata only).
    fl = q(con, """SELECT a.assigned_condition, c.in_control, c.in_treatment, count(*) n
                   FROM alogs a LEFT JOIN cond c USING (assignment_log_id) GROUP BY ALL ORDER BY ALL""")
    fl.to_csv(AUDIT / "labels_vs_flags.csv", index=False)
    S["labels_vs_flags"] = fl.to_dict("records")
    ncond = q(con, """SELECT experiment_id, count(DISTINCT assigned_condition) FILTER (WHERE assigned_condition <> 'Not Assigned') AS n_arms,
                             count(*) FILTER (WHERE assigned_condition <> 'Not Assigned') AS n_assigned,
                             count(*) FILTER (WHERE assigned_condition = 'Not Assigned') AS n_not_assigned
                      FROM alogs GROUP BY 1 ORDER BY 1""")
    ncond.to_csv(AUDIT / "experiments_arms.csv", index=False)
    S["experiments_by_n_arms"] = ncond.n_arms.value_counts().sort_index().to_dict()
    # Per experiment x arm sizes (assignment only).
    arms = q(con, """SELECT experiment_id, assigned_condition, count(*) n FROM alogs
                     WHERE assigned_condition <> 'Not Assigned' GROUP BY 1,2 ORDER BY 1,2""")
    arms.to_csv(AUDIT / "experiment_arm_sizes.csv", index=False)
    # Linkage alogs -> targets and student-id agreement; experiment_id vs target_sequence.
    r = one(con, """SELECT count(*), count(t.assignment_log_id),
                           sum(CASE WHEN t.assignment_log_id IS NOT NULL AND t.student_id <> a.student_id THEN 1 ELSE 0 END),
                           sum(CASE WHEN t.assignment_log_id IS NOT NULL AND t.target_sequence <> a.experiment_id THEN 1 ELSE 0 END)
                    FROM alogs a LEFT JOIN tgt t USING (assignment_log_id)""")
    S["alogs_to_targets"] = dict(zip(["alog_rows", "matched_targets", "student_mismatch", "experiment_vs_target_sequence_mismatch"],
                                     map(int, r)))
    S["targets_without_alog"] = int(one(con, "SELECT count(*) FROM tgt ANTI JOIN alogs USING (assignment_log_id)")[0])
    # Start time agreement (alog start_time is a local-time string; targets use epoch seconds).
    st = q(con, """SELECT epoch(CAST(a.start_time AS TIMESTAMPTZ)) - t.assignment_start_time AS d
                   FROM alogs a JOIN tgt t USING (assignment_log_id)""")
    S["alog_vs_target_start_seconds"] = st.d.abs().quantile([.5, .9, .99, 1]).to_dict()
    # Clustering keys from priors.csv.
    r = one(con, "SELECT count(DISTINCT class_id), count(DISTINCT teacher_id), count(DISTINCT district_id) FROM covp")
    S["cov_priors_clusters"] = dict(zip(["classes", "teachers", "districts"], map(int, r)))

    # ---- 3. Target table and linkage ------------------------------------------------------
    r = one(con, "SELECT count(*), count(DISTINCT assignment_log_id), count(DISTINCT student_id), "
                 "count(DISTINCT class_id), count(DISTINCT target_sequence) FROM tgt")
    S["tgt_rows"], S["tgt_distinct_alog"], S["tgt_distinct_student"], S["tgt_distinct_class"], \
        S["tgt_distinct_sequence"] = map(int, r)
    r = one(con, """SELECT count(*) AS n_tgt,
                           count(c.assignment_log_id) AS n_matched,
                           sum(CASE WHEN c.assignment_log_id IS NOT NULL AND c.user_id <> t.student_id THEN 1 ELSE 0 END)
                               AS id_mismatch
                    FROM tgt t LEFT JOIN cond c USING (assignment_log_id)""")
    S["tgt_rows_with_condition"], S["tgt_cond_student_id_mismatch"] = int(r[1]), int(r[2])
    S["cond_rows_without_target"] = int(one(con, "SELECT count(*) FROM cond c ANTI JOIN tgt t USING (assignment_log_id)")[0])

    # Linked randomized sample: target rows with exactly one valid flag.
    con.execute("""CREATE OR REPLACE TABLE rct AS
        SELECT t.*, CAST(c.in_treatment AS INTEGER) AS z
        FROM tgt t JOIN cond c USING (assignment_log_id)
        WHERE (c.in_control = 1 AND c.in_treatment = 0) OR (c.in_control = 0 AND c.in_treatment = 1)""")
    S["rct_rows"] = int(one(con, "SELECT count(*) FROM rct")[0])
    S["rct_students"] = int(one(con, "SELECT count(DISTINCT student_id) FROM rct")[0])
    S["rct_sequences"] = int(one(con, "SELECT count(DISTINCT target_sequence) FROM rct")[0])

    # Do target sequences of non-linked target rows look like non-randomized starts?
    seq = q(con, """SELECT t.target_sequence,
                           count(*) AS n_target_rows,
                           count(c.assignment_log_id) AS n_with_flags,
                           sum(CASE WHEN c.in_treatment=1 AND c.in_control=0 THEN 1 ELSE 0 END) AS n_treat,
                           sum(CASE WHEN c.in_control=1 AND c.in_treatment=0 THEN 1 ELSE 0 END) AS n_ctrl,
                           min(t.assignment_start_time) AS first_start, max(t.assignment_start_time) AS last_start
                    FROM tgt t LEFT JOIN cond c USING (assignment_log_id) GROUP BY 1 ORDER BY 2 DESC""")
    seq["first_start"] = pd.to_datetime(seq.first_start, unit="s").dt.date
    seq["last_start"] = pd.to_datetime(seq.last_start, unit="s").dt.date
    nm = q(con, "SELECT * FROM raw_exp_norm_map")
    seq["in_norm_map_experiment_id"] = seq.target_sequence.isin(set(nm.experiment_id))
    seq["share_treat"] = seq.n_treat / (seq.n_treat + seq.n_ctrl)
    seq.to_csv(AUDIT / "sequences_arm_sizes.csv", index=False)
    S["norm_map_rows"] = len(nm)
    S["norm_map_experiment_ids_nonnull"] = int(nm.experiment_id.notna().sum())
    S["norm_map_normal_ids_nonnull"] = int(nm.normal_id.notna().sum())
    S["sequences_with_both_arms"] = int(((seq.n_treat > 0) & (seq.n_ctrl > 0)).sum())

    # Same student, same target sequence more than once?
    S["rct_dup_student_sequence"] = int(one(con, "SELECT count(*) FROM (SELECT student_id, target_sequence FROM rct "
                                                 "GROUP BY 1,2 HAVING count(*)>1)")[0])
    rep = q(con, "SELECT student_id, count(*) n FROM rct GROUP BY 1")
    S["rct_rows_per_student"] = rep.n.value_counts().sort_index().to_dict()

    # ---- 4. Overall outcome availability (never by arm) -------------------------------------
    r = one(con, """SELECT count(*), count(assignment_completed), count(problems_completed),
                           min(assignment_completed), max(assignment_completed),
                           count(DISTINCT assignment_completed)
                    FROM rct""")
    S["outcome_availability"] = dict(zip(
        ["n", "completed_nonnull", "problems_nonnull", "completed_min", "completed_max", "completed_distinct"],
        [int(x) for x in r]))
    # Structural check of problems_completed versus completion status (outcome x outcome, arm-blind).
    pc = q(con, """SELECT assignment_completed, count(*) n, min(problems_completed) mn,
                          quantile_cont(problems_completed, 0.5) med, max(problems_completed) mx,
                          sum(CASE WHEN problems_completed IS NULL THEN 1 ELSE 0 END) n_null
                   FROM rct GROUP BY 1 ORDER BY 1""")
    pc.to_csv(AUDIT / "problems_completed_structure_armblind.csv", index=False)
    S["problems_completed_by_completion_armblind"] = pc.to_dict("records")

    # ---- 5. Students, classes, overlap, grouping structure ------------------------------
    S["remnant_targets_rows"] = int(one(con, "SELECT count(*) FROM rtgt")[0])
    S["overlap_students_rct_remnant"] = int(one(con, "SELECT count(*) FROM (SELECT DISTINCT student_id FROM rct) "
                                                     "SEMI JOIN (SELECT DISTINCT student_id FROM rtgt) USING (student_id)")[0])
    S["overlap_classes_rct_remnant"] = int(one(con, "SELECT count(*) FROM (SELECT DISTINCT class_id FROM rct) "
                                                    "SEMI JOIN (SELECT DISTINCT class_id FROM rtgt) USING (class_id)")[0])
    S["overlap_sequences_rct_remnant"] = int(one(con, "SELECT count(*) FROM (SELECT DISTINCT target_sequence FROM rct) "
                                                      "SEMI JOIN (SELECT DISTINCT target_sequence FROM rtgt) USING (target_sequence)")[0])
    pairs = q(con, "SELECT DISTINCT student_id, class_id FROM rct")
    S["rct_classes"] = int(pairs.class_id.nunique())
    S["students_per_class_quantiles"] = pairs.groupby("class_id").size().quantile([.1, .25, .5, .75, .9, 1]).to_dict()
    S["classes_with_one_rct_student"] = int((pairs.groupby("class_id").size() == 1).sum())
    S["classes_per_student_counts"] = pairs.groupby("student_id").size().value_counts().sort_index().to_dict()
    # Connected components of the bipartite student-class graph (union-find).
    parent = {}

    def find(x):
        while parent.setdefault(x, x) != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for s, c in zip(pairs.student_id, pairs.class_id):
        a, b = find(("s", s)), find(("c", c))
        if a != b:
            parent[a] = b
    comp_sizes = collections.Counter(find(("s", s)) for s in pairs.student_id.unique())
    sizes = np.array(sorted(comp_sizes.values(), reverse=True))
    S["student_class_components"] = {"n_components": int(len(sizes)), "largest_component_students": int(sizes[0]),
                                     "largest_share": float(sizes[0] / sizes.sum()),
                                     "top5": sizes[:5].tolist()}

    # ---- 6. Temporal integrity of histories ---------------------------------------------
    with Timer("history joins", log):
        con.execute("""CREATE OR REPLACE TABLE hist_counts AS
            SELECT r.assignment_log_id,
                   count(a.student_id) FILTER (WHERE a.assignment_start_time <  r.assignment_start_time) AS n_prior,
                   count(a.student_id) FILTER (WHERE a.assignment_start_time =  r.assignment_start_time) AS n_equal,
                   count(a.student_id) FILTER (WHERE abs(a.assignment_start_time - r.assignment_start_time) < 1
                                                 AND a.assignment_start_time <> r.assignment_start_time) AS n_within_1s,
                   count(a.student_id) FILTER (WHERE a.assignment_start_time >  r.assignment_start_time) AS n_after,
                   count(a.student_id) FILTER (WHERE a.assignment_start_time < r.assignment_start_time
                                                 AND a.average_problem_correctness IS NOT NULL) AS n_prior_with_stats,
                   max(a.assignment_start_time) FILTER (WHERE a.assignment_start_time < r.assignment_start_time) AS last_prior_start
            FROM rct r LEFT JOIN raw_exp_assign a ON a.student_id = r.student_id
            GROUP BY 1""")
    hc = q(con, """SELECT h.*, r.student_prior_assignments_started AS prior_started_feature,
                          r.assignment_start_time AS target_start
                   FROM hist_counts h JOIN rct r USING (assignment_log_id)""")
    S["history"] = {
        "rows": int(len(hc)),
        "any_equal_start": int((hc.n_equal > 0).sum()),
        "any_within_1s": int((hc.n_within_1s > 0).sum()),
        "rows_with_future_history_rows": int((hc.n_after > 0).sum()),
        "n_prior_quantiles": hc.n_prior.quantile([0, .1, .25, .5, .75, .9, 1]).to_dict(),
        "n_prior_with_stats_quantiles": hc.n_prior_with_stats.quantile([0, .1, .25, .5, .75, .9, 1]).to_dict(),
        "share_n_prior_with_stats_ge": {k: float((hc.n_prior_with_stats >= k).mean()) for k in (1, 5, 10, 20)},
        # Do precomputed 'prior assignments started' agree with the raw history count?
        "prior_started_feature_vs_raw": {
            "nonnull": int(hc.prior_started_feature.notna().sum()),
            "exact_equal_share": float((hc.prior_started_feature == hc.n_prior).mean()),
            "spearman": float(hc[["prior_started_feature", "n_prior"]].corr("spearman").iloc[0, 1]),
            "median_ratio": float((hc.prior_started_feature / hc.n_prior.replace(0, np.nan)).median()),
        },
    }
    hc[["n_prior", "n_prior_with_stats"]].describe().to_csv(AUDIT / "history_length_describe.csv")

    with Timer("action joins", log):
        ac = q(con, """SELECT
                count(*) FILTER (WHERE x.timestamp + 86400 <= r.assignment_start_time) AS days_strictly_before,
                count(*) FILTER (WHERE x.timestamp <= r.assignment_start_time
                                   AND x.timestamp + 86400 > r.assignment_start_time) AS target_day,
                count(*) FILTER (WHERE x.timestamp > r.assignment_start_time) AS after,
                count(DISTINCT r.assignment_log_id) FILTER (WHERE x.timestamp + 86400 <= r.assignment_start_time
                                   AND x.timestamp >= r.assignment_start_time - 61*86400) AS targets_with_actions_last60
            FROM rct r JOIN raw_exp_actions x ON x.student_id = r.student_id""")
    S["actions"] = {k: int(v) for k, v in ac.iloc[0].items()}
    S["actions"]["timestamps_midnight_utc_share"] = float(one(
        con, "SELECT avg(CASE WHEN timestamp % 86400 = 0 THEN 1 ELSE 0 END) FROM raw_exp_actions")[0])

    # ---- 7. Eligibility and retention per contrast (arm sizes only) -----------------------
    con.execute("CREATE OR REPLACE TABLE rct_hist AS SELECT r.*, h.n_prior, h.n_prior_with_stats "
                "FROM rct r JOIN hist_counts h USING (assignment_log_id)")
    elig = q(con, """SELECT target_sequence,
                count(*) FILTER (WHERE z=1) n_treat_all, count(*) FILTER (WHERE z=0) n_ctrl_all,
                count(*) FILTER (WHERE z=1 AND n_prior_with_stats>=10) n_treat_h10,
                count(*) FILTER (WHERE z=0 AND n_prior_with_stats>=10) n_ctrl_h10,
                count(*) FILTER (WHERE z=1 AND n_prior_with_stats>=5) n_treat_h5,
                count(*) FILTER (WHERE z=0 AND n_prior_with_stats>=5) n_ctrl_h5
            FROM rct_hist GROUP BY 1 ORDER BY 1""")
    elig.to_csv(AUDIT / "eligibility_by_sequence.csv", index=False)
    for tag in ("all", "h10", "h5"):
        ok = elig[(elig[f"n_treat_{tag}"] >= 56) & (elig[f"n_ctrl_{tag}"] >= 56)]
        ok20 = elig[(elig[f"n_treat_{tag}"] >= 20) & (elig[f"n_ctrl_{tag}"] >= 20)]
        S[f"eligible_{tag}"] = {
            "rows_total": int(elig[f"n_treat_{tag}"].sum() + elig[f"n_ctrl_{tag}"].sum()),
            "sequences_ge56_per_arm": int(len(ok)),
            "rows_in_ge56": int(ok[f"n_treat_{tag}"].sum() + ok[f"n_ctrl_{tag}"].sum()),
            "sequences_ge20_per_arm": int(len(ok20)),
            "rows_in_ge20": int(ok20[f"n_treat_{tag}"].sum() + ok20[f"n_ctrl_{tag}"].sum()),
        }
    # Arm-share descriptive (not used as a filter).
    elig["share_treat_all"] = elig.n_treat_all / (elig.n_treat_all + elig.n_ctrl_all)
    S["arm_share_all_quantiles"] = elig.share_treat_all.quantile([0, .1, .25, .5, .75, .9, 1]).to_dict()

    # Exclusion by arm under the history rule (counts only).
    ex = q(con, """SELECT z, count(*) n, avg(CASE WHEN n_prior_with_stats>=10 THEN 1.0 ELSE 0 END) share_h10,
                          avg(CASE WHEN n_prior_with_stats>=5 THEN 1.0 ELSE 0 END) share_h5
                   FROM rct_hist GROUP BY 1 ORDER BY 1""")
    ex.to_csv(AUDIT / "history_eligibility_by_arm.csv", index=False)
    S["history_eligibility_by_arm"] = ex.to_dict("records")

    # Start-date range of randomized targets.
    r = one(con, "SELECT min(assignment_start_time), max(assignment_start_time) FROM rct")
    S["rct_date_range"] = [str(pd.to_datetime(r[0], unit="s")), str(pd.to_datetime(r[1], unit="s"))]

    save_json(S, AUDIT / "audit_summary.json")
    log("audit complete")


if __name__ == "__main__":
    main()
