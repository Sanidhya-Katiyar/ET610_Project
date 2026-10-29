"""Outcome-blind feasibility audit of post-target ("follow-up") rows for a downstream-effects study.

No outcome of any kind is tabulated by arm. Arm appears only in pre-treatment eligibility checks
(target calendar date). Outputs aggregates only:
  reports/audit_tables/followup_feasibility.json
  reports/audit_tables/followup_window_by_contrast_sample.csv
"""
import numpy as np
import pandas as pd

from common import AUDIT, DERIVED, connect, create_raw_views, make_logger, save_json

log = make_logger("10_followup_feasibility")
DAY = 86400.0


def main():
    con = connect(read_only=False)
    create_raw_views(con)
    q = lambda s: con.execute(s).df()
    res = {}

    # 1. Extraction rule: is any history / action row later than the student's last experimental target?
    for name, view, tcol in [("history", "raw_exp_assign", "assignment_start_time"), ("actions", "raw_exp_actions", "timestamp")]:
        d = q(f"""WITH h AS (SELECT student_id, max({tcol}) hmax FROM {view} GROUP BY 1),
                       t AS (SELECT student_id, max(assignment_start_time) tmax FROM raw_exp_targets GROUP BY 1)
                  SELECT count(*) students,
                         sum(CASE WHEN hmax > tmax + 1 THEN 1 ELSE 0 END) students_with_rows_after_last_target,
                         max((hmax - tmax) / {DAY}) max_days_after_last_target,
                         quantile_cont((hmax - tmax) / {DAY}, 0.5) median_days_last_row_minus_last_target
                  FROM h JOIN t USING (student_id)""").iloc[0]
        res[f"{name}_extraction"] = {k: float(v) for k, v in d.items()}
    # Same check for the remnant (does its history stop at the remnant target?).
    d = q(f"""WITH h AS (SELECT student_id, max(assignment_start_time) hmax FROM raw_rem_assign GROUP BY 1),
                   t AS (SELECT student_id, max(assignment_start_time) tmax FROM raw_rem_targets GROUP BY 1)
              SELECT count(*) students, sum(CASE WHEN hmax > tmax + 1 THEN 1 ELSE 0 END) after_target
              FROM h JOIN t USING (student_id)""").iloc[0]
    res["remnant_extraction"] = {k: float(v) for k, v in d.items()}

    # 2. Follow-up window per randomized unit = time until the student's LAST experimental target start
    #    (the extraction end). Arm-blind.
    con.execute(f"CREATE OR REPLACE TEMP VIEW units AS SELECT * FROM read_parquet('{(DERIVED / 'units_exp.parquet').as_posix()}')")
    w = q(f"""WITH t AS (SELECT student_id, max(assignment_start_time) tmax FROM raw_exp_targets GROUP BY 1)
              SELECT u.unit_id, u.experiment_id, (t.tmax - u.assignment_start_time) / {DAY} AS window_days
              FROM units u JOIN t USING (student_id)""")
    res["units"] = int(len(w))
    res["window_days_quantiles"] = {str(p): float(w.window_days.quantile(p)) for p in (0.1, 0.25, 0.5, 0.75, 0.9)}
    for k in (1, 7, 14, 30):
        res[f"share_units_window_ge_{k}d"] = float((w.window_days >= k).mean())
    res["share_units_no_window"] = float((w.window_days <= 1e-6).mean())

    # 3. What ends the window: the student's later experimental target. Count later targets per unit.
    nxt = q(f"""SELECT u.unit_id, min(t.assignment_start_time - u.assignment_start_time) / {DAY} AS days_to_next_target,
                       count(t.assignment_start_time) AS n_later_targets
                FROM units u LEFT JOIN raw_exp_targets t
                  ON t.student_id = u.student_id AND t.assignment_start_time > u.assignment_start_time + 1
                GROUP BY 1""")
    res["share_units_with_later_target"] = float(nxt.n_later_targets.gt(0).mean())
    res["days_to_next_target_quantiles"] = {str(p): float(nxt.days_to_next_target.quantile(p)) for p in (0.1, 0.25, 0.5, 0.75, 0.9)}

    # 4. Primary-sample view: per contrast, share of units with >= 7 / 30 days of follow-up (arm-blind).
    rows = pd.read_parquet(DERIVED / "contrasts.parquet", columns=["contrast_id", "unit_id"])
    ws = rows.merge(w[["unit_id", "window_days"]], on="unit_id")
    by_c = ws.groupby("contrast_id").window_days.agg(n="size", share_ge7=lambda x: (x >= 7).mean(),
                                                     share_ge30=lambda x: (x >= 30).mean())
    by_c.to_csv(AUDIT / "followup_window_by_contrast_sample.csv")
    res["contrasts"] = int(len(by_c))
    res["contrasts_with_ge100_units_7d"] = int((by_c.n * by_c.share_ge7 >= 100).sum())
    res["rows_ge7d"] = int((ws.window_days >= 7).sum())
    res["rows_ge30d"] = int((ws.window_days >= 30).sum())

    # 5. Pooled, arm-blind: how many history rows fall inside the window, by days after target.
    cnt = q(f"""WITH t AS (SELECT student_id, max(assignment_start_time) tmax FROM raw_exp_targets GROUP BY 1)
                SELECT (a.assignment_start_time - u.assignment_start_time) / {DAY} AS d
                FROM units u JOIN raw_exp_assign a ON a.student_id = u.student_id
                 AND a.assignment_start_time > u.assignment_start_time + 1
                 AND a.assignment_start_time <= u.assignment_start_time + 30 * {DAY}""")
    res["future_rows_within_30d_total"] = int(len(cnt))
    res["future_rows_within_7d_total"] = int((cnt.d <= 7).sum())

    save_json(res, AUDIT / "followup_feasibility.json")
    log(f"follow-up feasibility: {res}")
    for k, v in res.items():
        print(k, v)


if __name__ == "__main__":
    main()
