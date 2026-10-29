"""Feature construction with explicit, testable time cutoffs.

All features are computed ONLY from history rows that satisfy
    history.assignment_start_time <= unit.assignment_start_time - buffer_s
(primary buffer_s = 86400, i.e. assignments started at least 24 h before the target start),
and from daily action counts whose day ended at least `buffer_s` before the target start
(timestamp + 86400 <= target_start - (buffer_s - 86400)); with buffer_s = 86400 this is the
authors' own rule `timestamp + 86400 < start`, made non-strict.

Level features (X_B) summarise volume, recency and current performance level.
Trajectory features (X_T) summarise change, shape and variability within the last K
prior assignments that have problem-level statistics.
"""
import numpy as np
import pandas as pd

LEVEL_FEATURES = [
    "b_log_n_prior",            # log(1 + prior assignments started)
    "b_log_n_prior_stats",      # log(1 + prior assignments with problem statistics)
    "b_log_tenure_days",        # log(1 + days since first recorded prior assignment)
    "b_log_recency_days",       # log(1 + days since most recent eligible prior assignment start)
    "b_completion_rate_all",    # lifetime share of prior assignments completed
    "b_correct_all",            # lifetime mean first-attempt correctness (assignment-level means)
    "b_correct_norm_win",       # mean difficulty-normalized correctness over the window
    "b_ln_time_on_task_win",    # mean of per-assignment median ln problem time on task (window)
    "b_ln_first_resp_win",      # mean of per-assignment median ln first response time (window)
    "b_attempts_win",           # mean attempts per problem (window)
    "b_hints_win",              # mean hints per problem (window)
    "b_answer_given_win",       # mean share of problems where the answer was given (window)
    "b_problems_per_assign_win",  # mean problems completed per assignment (window)
    "b_skill_builder_share_win",  # share of window assignments that are skill builders
    "b_active_days_60",         # number of active days in the 60 days before the cutoff
    "b_has_due_date",           # target assignment has a due date (assignment setting, pre-treatment)
]

TRAJECTORY_FEATURES = [
    "t_slope_correct_norm",     # OLS slope of normalized correctness on chronological index (per assignment)
    "t_resid_sd_correct_norm",  # residual SD around that slope
    "t_recent_shift_correct",   # mean normalized correctness of last 5 minus preceding (up to 15)
    "t_completion_shift",       # completion rate of last 5 minus preceding (up to 15)
    "t_slope_hints_norm",       # OLS slope of normalized hint count on chronological index
    "t_activity_trend",         # active days in days 1-30 minus days 31-60 before the cutoff
    "t_gap_cv",                 # coefficient of variation of log inter-start gaps within the window
]


def prior_history_sql(units, history, buffer_s):
    """SQL selecting eligible prior history rows for each unit, with recency ranks.

    units:   table/view with columns unit_id, student_id, assignment_start_time
    history: table/view with assignment-level history (raw schema)
    Returns columns: unit_id, h_start, has_stats, rn_all (1 = most recent), rn_stats, plus stats.
    """
    return f"""
    SELECT u.unit_id, a.assignment_start_time AS h_start,
           a.average_problem_correctness IS NOT NULL AS has_stats,
           a.assignment_completed, a.is_skill_builder, a.time_since_last_assignment_start,
           a.completed_problem_count_raw, a.median_ln_problem_time_on_task_raw,
           a.median_ln_problem_first_response_time_raw, a.average_problem_attempt_count,
           a.average_problem_correctness, a.average_problem_correctness_normalized,
           a.average_problem_hint_count, a.average_problem_hint_count_normalized,
           a.average_problem_answer_given,
           row_number() OVER (PARTITION BY u.unit_id ORDER BY a.assignment_start_time DESC) AS rn_all
    FROM {units} u JOIN {history} a
      ON a.student_id = u.student_id
     AND a.assignment_start_time <= u.assignment_start_time - {float(buffer_s)}
    """


def lifetime_sql(prior):
    """Lifetime (all eligible prior rows) aggregates per unit."""
    return f"""
    SELECT unit_id,
           count(*) AS n_prior,
           count(*) FILTER (WHERE has_stats) AS n_prior_stats,
           min(h_start) AS first_start, max(h_start) AS last_start,
           avg(assignment_completed) AS b_completion_rate_all,
           avg(average_problem_correctness) AS b_correct_all
    FROM {prior} GROUP BY unit_id
    """


def actions_sql(units, actions, buffer_s):
    """Active-day counts from daily action files (rows exist only for days with activity)."""
    cut = f"(u.assignment_start_time - {float(buffer_s) - 86400.0})"
    return f"""
    SELECT u.unit_id,
           count(*) FILTER (WHERE x.timestamp + 86400 > {cut} - 60*86400) AS b_active_days_60,
           count(*) FILTER (WHERE x.timestamp + 86400 > {cut} - 30*86400) AS act_1_30,
           count(*) FILTER (WHERE x.timestamp + 86400 <= {cut} - 30*86400
                              AND x.timestamp + 86400 > {cut} - 60*86400) AS act_31_60
    FROM {units} u JOIN {actions} x
      ON x.student_id = u.student_id
     AND x.timestamp + 86400 <= {cut}
     AND x.timestamp + 86400 > {cut} - 60*86400
    GROUP BY u.unit_id
    """


def _ols_slope_resid(y, x):
    """Per-group OLS slope and residual SD for equal-length arrays; NaN if < 3 points."""
    n = len(y)
    if n < 3:
        return np.nan, np.nan
    xm, ym = x.mean(), y.mean()
    sxx = ((x - xm) ** 2).sum()
    if sxx == 0:
        return np.nan, np.nan
    b = ((x - xm) * (y - ym)).sum() / sxx
    resid = y - ym - b * (x - xm)
    return b, np.sqrt((resid ** 2).sum() / (n - 2))


def window_features(win, K):
    """Compute window-based level and trajectory features.

    win: DataFrame of prior rows WITH stats for each unit, columns unit_id, h_start,
         assignment_completed, is_skill_builder, completed_problem_count_raw,
         median_ln_problem_time_on_task_raw, median_ln_problem_first_response_time_raw,
         average_problem_attempt_count, average_problem_correctness_normalized,
         average_problem_hint_count, average_problem_hint_count_normalized,
         average_problem_answer_given. Only the last K rows (by h_start) per unit are used.
    """
    win = win.sort_values(["unit_id", "h_start"])
    win = win.groupby("unit_id", sort=False).tail(K)
    out = {}
    for uid, g in win.groupby("unit_id", sort=False):
        n = len(g)
        x = np.arange(n, dtype=float)
        yc = g["average_problem_correctness_normalized"].to_numpy(float)
        yh = g["average_problem_hint_count_normalized"].to_numpy(float)
        comp = g["assignment_completed"].to_numpy(float)
        okc, okh = ~np.isnan(yc), ~np.isnan(yh)
        slope_c, rsd_c = _ols_slope_resid(yc[okc], x[okc])
        slope_h, _ = _ols_slope_resid(yh[okh], x[okh])
        if n >= 6:
            last, prev = slice(n - 5, n), slice(0, n - 5)
            shift_c = np.nanmean(yc[last]) - np.nanmean(yc[prev])
            shift_comp = comp[last].mean() - comp[prev].mean()
        else:
            shift_c = shift_comp = np.nan
        # Gap regularity: CV of log(1 + gap in hours) between consecutive window starts.
        gaps = np.diff(g["h_start"].to_numpy(float)) / 3600.0
        lg = np.log1p(np.clip(gaps, 0, None))
        gap_cv = lg.std(ddof=1) / lg.mean() if len(lg) >= 3 and lg.mean() > 0 else np.nan
        out[uid] = {
            "n_win": n,
            "b_correct_norm_win": np.nanmean(yc) if okc.any() else np.nan,
            "b_ln_time_on_task_win": np.nanmean(g["median_ln_problem_time_on_task_raw"].to_numpy(float)),
            "b_ln_first_resp_win": np.nanmean(g["median_ln_problem_first_response_time_raw"].to_numpy(float)),
            "b_attempts_win": np.nanmean(g["average_problem_attempt_count"].to_numpy(float)),
            "b_hints_win": np.nanmean(g["average_problem_hint_count"].to_numpy(float)),
            "b_answer_given_win": np.nanmean(g["average_problem_answer_given"].to_numpy(float)),
            "b_problems_per_assign_win": np.nanmean(g["completed_problem_count_raw"].to_numpy(float)),
            "b_skill_builder_share_win": np.nanmean(g["is_skill_builder"].to_numpy(float)),
            "t_slope_correct_norm": slope_c,
            "t_resid_sd_correct_norm": rsd_c,
            "t_recent_shift_correct": shift_c,
            "t_completion_shift": shift_comp,
            "t_slope_hints_norm": slope_h,
            "t_gap_cv": gap_cv,
        }
    return pd.DataFrame.from_dict(out, orient="index").rename_axis("unit_id").reset_index()


def assemble(units, life, acts, win):
    """Join pieces into one feature frame (one row per unit)."""
    df = units[["unit_id", "assignment_start_time", "has_due_date"]].merge(life, on="unit_id", how="left")
    df = df.merge(acts, on="unit_id", how="left").merge(win, on="unit_id", how="left")
    for c in ("n_prior", "n_prior_stats", "b_active_days_60", "act_1_30", "act_31_60", "n_win"):
        df[c] = df[c].fillna(0)
    df["b_log_n_prior"] = np.log1p(df.n_prior)
    df["b_log_n_prior_stats"] = np.log1p(df.n_prior_stats)
    df["b_log_tenure_days"] = np.log1p((df.assignment_start_time - df.first_start) / 86400.0)
    df["b_log_recency_days"] = np.log1p((df.assignment_start_time - df.last_start) / 86400.0)
    df["b_has_due_date"] = df.has_due_date.astype(float)
    df["t_activity_trend"] = df.act_1_30 - df.act_31_60
    return df
