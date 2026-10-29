"""Unit tests on small synthetic fixtures (no real data needed).

Run: .venv/Scripts/python -m pytest -q tests
"""
import importlib.util
import pathlib
import sys

import duckdb
import numpy as np
import pandas as pd
import pytest

SRC = pathlib.Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC))
import estimation as E  # noqa: E402
import features as F  # noqa: E402
import pipeline as P  # noqa: E402


def _load(name, file):
    spec = importlib.util.spec_from_file_location(name, SRC / file)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


DAY = 86400.0
T0 = 1_600_000_000.0  # target start


def _history_fixture():
    """One unit; history rows at -10d, -2d, -23h, -1s, 0 (the target itself), +1d."""
    offs = [-10 * DAY, -2 * DAY, -23 * 3600, -1, 0, DAY]
    hist = pd.DataFrame({
        "student_id": 1, "assignment_start_time": [T0 + o for o in offs],
        "assignment_completed": [1, 0, 1, 1, 1, 1], "is_skill_builder": 1, "time_since_last_assignment_start": 1.0,
        "completed_problem_count_raw": 5, "median_ln_problem_time_on_task_raw": 3.0,
        "median_ln_problem_first_response_time_raw": 2.0, "average_problem_attempt_count": 1.2,
        "average_problem_correctness": [0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
        "average_problem_correctness_normalized": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
        "average_problem_hint_count": 0.1, "average_problem_hint_count_normalized": 0.0,
        "average_problem_answer_given": 0.0,
    })
    units = pd.DataFrame({"unit_id": [7], "student_id": [1], "assignment_start_time": [T0], "has_due_date": [0]})
    return hist, units


def test_prior_rows_respect_24h_buffer():
    hist, units = _history_fixture()
    con = duckdb.connect()
    con.register("h", hist)
    con.register("u", units)
    prior = con.execute(F.prior_history_sql("u", "h", 86400)).df()
    assert len(prior) == 2                       # only -10d and -2d
    assert (prior.h_start <= T0 - DAY).all()
    strict = con.execute(F.prior_history_sql("u", "h", 60)).df()
    assert len(strict) == 3                      # -1s, the target itself and the future row are excluded
    assert (strict.h_start < T0 - 59).all()


def test_actions_exclude_target_day_and_future():
    con = duckdb.connect()
    day0 = np.floor(T0 / DAY) * DAY  # midnight UTC of the target day
    acts = pd.DataFrame({"student_id": 1, "timestamp": [day0 - 45 * DAY, day0 - 10 * DAY, day0 - DAY, day0, day0 + DAY],
                         "ln_action_1_count": 0.69})
    units = pd.DataFrame({"unit_id": [7], "student_id": [1], "assignment_start_time": [T0], "has_due_date": [0]})
    con.register("a", acts)
    con.register("u", units)
    r = con.execute(F.actions_sql("u", "a", 86400)).df().iloc[0]
    # Authors' rule: a calendar day counts only if it ended at or before the target start
    # (-45d, -10d and the day before the target); the target day and later days are excluded.
    assert r.b_active_days_60 == 3
    assert r.act_1_30 == 2 and r.act_31_60 == 1


def test_window_features_slope_shift_and_K():
    n = 12
    win = pd.DataFrame({
        "unit_id": 1, "h_start": np.arange(n) * DAY, "assignment_completed": [0] * 7 + [1] * 5,
        "is_skill_builder": 1, "completed_problem_count_raw": 4.0, "median_ln_problem_time_on_task_raw": 3.0,
        "median_ln_problem_first_response_time_raw": 2.0, "average_problem_attempt_count": 1.0,
        "average_problem_correctness_normalized": 0.5 * np.arange(n) + 1.0,
        "average_problem_hint_count": 0.0, "average_problem_hint_count_normalized": -0.1 * np.arange(n),
        "average_problem_answer_given": 0.0,
    })
    out = F.window_features(win, K=10).iloc[0]
    assert out.n_win == 10
    assert out.t_slope_correct_norm == pytest.approx(0.5)
    assert out.t_resid_sd_correct_norm == pytest.approx(0.0, abs=1e-12)
    assert out.t_slope_hints_norm == pytest.approx(-0.1)
    # last 10 rows are indices 2..11; last 5 = 7..11 (mean 10), previous 5 = 2..6 (mean 3.5 + ... )
    vals = 0.5 * np.arange(2, 12) + 1.0
    assert out.t_recent_shift_correct == pytest.approx(vals[5:].mean() - vals[:5].mean())
    assert out.t_completion_shift == pytest.approx(1.0 - 0.0)


def test_gap_cv_constant_gaps_is_zero_or_nan():
    win = pd.DataFrame({"unit_id": 1, "h_start": np.arange(6) * DAY, "assignment_completed": 1, "is_skill_builder": 1,
                        "completed_problem_count_raw": 1.0, "median_ln_problem_time_on_task_raw": 1.0,
                        "median_ln_problem_first_response_time_raw": 1.0, "average_problem_attempt_count": 1.0,
                        "average_problem_correctness_normalized": np.arange(6.0), "average_problem_hint_count": 0.0,
                        "average_problem_hint_count_normalized": 0.0, "average_problem_answer_given": 0.0})
    assert F.window_features(win, 20).iloc[0].t_gap_cv == pytest.approx(0.0)


def test_contrasts_orientation_and_membership():
    mod = _load("build", "02_build_sample.py")
    units = pd.DataFrame({"unit_id": range(6), "experiment_id": "E1",
                          "arm": ["Control", "Treatment 1", "Treatment 2", "Control", "Treatment 1", "Treatment 2"]})
    c = mod.build_contrasts(units)
    assert set(c.contrast_id) == {"E1|Control|Treatment 1", "E1|Control|Treatment 2", "E1|Treatment 1|Treatment 2"}
    sub = c[c.contrast_id == "E1|Control|Treatment 2"]
    assert sorted(sub.unit_id) == [0, 2, 3, 5]
    assert (sub.set_index("unit_id").z.loc[[0, 3]] == 0).all() and (sub.set_index("unit_id").z.loc[[2, 5]] == 1).all()


def test_folds_never_split_groups():
    rng = np.random.default_rng(0)
    groups = rng.integers(0, 300, size=5000)
    folds = E.make_folds(groups, 5, seed=1)
    per_group = pd.Series(folds).groupby(groups).nunique()
    assert (per_group == 1).all()
    counts = np.bincount(folds)
    assert counts.min() > 0.8 * counts.max()


def test_dr_score_unbiased_with_wrong_nuisance():
    rng = np.random.default_rng(1)
    n = 400_000
    x = rng.normal(size=n)
    p = 0.3
    z = rng.binomial(1, p, n)
    tau = 0.1 + 0.05 * x
    y0 = rng.binomial(1, 0.5, n)
    y1 = rng.binomial(1, np.clip(0.5 + tau, 0, 1), n)
    y = np.where(z == 1, y1, y0)
    phi = E.dr_score(y, z, p, mu0=np.full(n, 0.9), mu1=np.full(n, 0.1))  # deliberately wrong nuisances
    assert phi.mean() == pytest.approx(tau.mean(), abs=0.01)
    # Loss differences rank estimators by tau-risk: true tau must beat a constant.
    d = (phi - 0.1) ** 2 - (phi - tau) ** 2
    true_gap = np.mean((tau - 0.1) ** 2)
    assert d.mean() == pytest.approx(true_gap, abs=0.004)


def test_pooled_ridge_recovers_contrast_specific_slopes():
    rng = np.random.default_rng(2)
    n_c, per = 4, 5000
    codes = np.repeat(np.arange(n_c), per)
    X = rng.normal(size=(n_c * per, 2))
    slopes = np.array([[1.0, 0], [-1.0, 0], [0.5, 0], [0, 0]])
    phi = 0.2 * codes + (X * slopes[codes]).sum(1) + rng.normal(scale=0.5, size=len(codes))
    m = E.PooledRidgeCATE(lam1=10, lam2=10).fit(X, phi, codes, n_c)
    np.testing.assert_allclose(m.beta + m.b, slopes, atol=0.05)
    const = E.PooledRidgeCATE().fit(X, phi, codes, n_c)
    np.testing.assert_allclose(const.predict(X, codes)[::per], m.alpha[np.arange(n_c)], atol=1e-9)


def test_cluster_se_matches_iid_when_singletons():
    rng = np.random.default_rng(3)
    d = rng.normal(size=1000)
    m, se = E.cluster_mean_se(d, np.arange(1000))
    assert se == pytest.approx(d.std(ddof=1) / np.sqrt(1000), rel=1e-6)


def test_pipeline_runs_and_detects_strong_heterogeneity():
    rng = np.random.default_rng(4)
    n = 6000
    df = pd.DataFrame({"contrast_id": rng.choice(["a", "b", "c"], n), "group": rng.integers(0, 1500, n),
                       "xb": rng.normal(size=n), "xt": rng.normal(size=n), "s": rng.normal(size=n)})
    df["z"] = rng.binomial(1, 0.5, n)
    tau = 0.3 * np.tanh(df.xt)
    p0 = 1 / (1 + np.exp(-(0.5 + 0.5 * df.s)))
    df["y"] = rng.binomial(1, np.clip(p0 + df.z * tau, 0.01, 0.99))
    out, _ = P.crossfit(df, ["xb"], ["xt"], seed=5)
    est, se = P.risk_diff(out, df, "MB", "MBT")
    assert est - 1.96 * se > 0
    assert out.phi.notna().all()


def test_outcome_join_is_one_to_one():
    mod = _load("analysis", "05_primary_analysis.py")
    rows = pd.DataFrame({"unit_id": [1, 1, 2], "contrast_id": ["a", "b", "a"], "z": [0, 1, 1]})
    outcomes = pd.DataFrame({"unit_id": [1, 2], "assignment_completed": [1, 0]})
    j = mod.join_outcomes(rows, outcomes)
    assert list(j.y) == [1, 1, 0]
    with pytest.raises(AssertionError):
        mod.join_outcomes(rows, pd.concat([outcomes, outcomes]))


def test_naive_buffers_leak_target_and_future_rows():
    """The naive variants used in src/14 (buffer 0 / huge negative) DO include the target row / future rows."""
    hist, units = _history_fixture()
    con = duckdb.connect()
    con.register("h", hist)
    con.register("u", units)
    self_leak = con.execute(F.prior_history_sql("u", "h", 0.0)).df()
    assert len(self_leak) == 5 and (self_leak.h_start == T0).any()      # includes the target itself
    all_leak = con.execute(F.prior_history_sql("u", "h", -1e10)).df()
    assert len(all_leak) == 6 and (all_leak.h_start > T0).any()         # includes the future row
