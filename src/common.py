"""Shared paths, DuckDB views and small helpers.

All raw CSVs are exposed as DuckDB views; nothing is loaded fully into pandas.
"""
import json
import pathlib
import time

import duckdb

ROOT = pathlib.Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw" / "osf_k8ph9" / "data"
DERIVED = ROOT / "data" / "derived"
REPORTS = ROOT / "reports"
FIGURES = ROOT / "figures"
LOGS = ROOT / "logs"
AUDIT = REPORTS / "audit_tables"
for d in (DERIVED, REPORTS, FIGURES, LOGS, AUDIT):
    d.mkdir(parents=True, exist_ok=True)

DB_PATH = DERIVED / "work.duckdb"

RAW_FILES = {
    "exp_conditions": "experiment_conditions.csv",
    "exp_targets": "experiment_prior_stats_and_targets.csv",
    "exp_assign": "experiment_assignment_stats.csv",
    "exp_actions": "experiment_action_counts.csv",
    "rem_targets": "remnant_prior_stats_and_targets.csv",
    "rem_assign": "remnant_assignment_stats.csv",
    "rem_actions": "remnant_action_counts.csv",
    "exp_norm_map": "exp_norm_map.csv",
}

# Problem-level statistics columns of the assignment-history files (verified against the CSV header).
ASSIGN_STAT_PREFIXES = [
    "session_count", "day_count", "completed_problem_count", "median_ln_problem_time_on_task",
    "median_ln_problem_first_response_time", "average_problem_attempt_count", "average_problem_answer_first",
    "average_problem_correctness", "average_problem_hint_count", "average_problem_answer_given",
]

SEED = 20261029


def connect(read_only=False, threads=8, memory_limit="6GB"):
    con = duckdb.connect(str(DB_PATH), read_only=read_only)
    con.execute(f"SET threads={threads}; SET memory_limit='{memory_limit}'; SET preserve_insertion_order=false; SET enable_progress_bar=false;")
    con.execute(f"SET temp_directory='{(DERIVED / 'duckdb_tmp').as_posix()}'")
    return con


def create_raw_views(con):
    """Expose each raw CSV as a view `raw_<name>`. The leading unnamed index column of
    experiment_conditions.csv is dropped."""
    for name, fname in RAW_FILES.items():
        path = (RAW / fname).as_posix()
        if not (RAW / fname).exists():
            raise FileNotFoundError(path)
        con.execute(
            f"CREATE OR REPLACE VIEW raw_{name} AS SELECT * FROM read_csv('{path}', header=true, "
            f"auto_detect=true, sample_size=-1)"
        )
    con.execute("CREATE OR REPLACE VIEW raw_exp_conditions2 AS SELECT * EXCLUDE (column0) FROM raw_exp_conditions")
    # Per-experiment E-TRIALS files from covariates.zip (extracted to RAW.parent / 'covariates').
    # Only assignment metadata columns are exposed: in-experiment performance columns are post-treatment
    # and outcome-like, and are deliberately not selected.
    cov = (RAW.parent / "covariates").as_posix()
    con.execute(f"""CREATE OR REPLACE VIEW raw_alogs AS
        SELECT experiment_id, CAST(student_id AS BIGINT) AS student_id, start_time,
               regexp_replace(trim(assigned_condition), '\\s+', ' ', 'g') AS assigned_condition,
               assigned_condition AS assigned_condition_raw,
               CAST(assistments_reference_assignment_log_id AS BIGINT) AS assignment_log_id
        FROM read_csv('{cov}/*/exp_alogs.csv', union_by_name=true, all_varchar=true)""")
    con.execute(f"""CREATE OR REPLACE VIEW raw_cov_priors AS
        SELECT experiment_id, CAST(student_id AS BIGINT) AS student_id, CAST(class_id AS BIGINT) AS class_id,
               CAST(teacher_id AS BIGINT) AS teacher_id, CAST(district_id AS BIGINT) AS district_id
        FROM read_csv('{cov}/*/priors.csv', union_by_name=true, all_varchar=true)""")


class Timer:
    def __init__(self, label, log):
        self.label, self.log = label, log

    def __enter__(self):
        self.t = time.time()

    def __exit__(self, *a):
        self.log(f"[{self.label}] {time.time() - self.t:.1f}s")


def make_logger(name):
    path = LOGS / f"{name}.log"
    fh = open(path, "a", encoding="utf-8")

    def log(msg):
        line = f"{time.strftime('%Y-%m-%d %H:%M:%S')} {msg}"
        print(line, flush=True)
        fh.write(line + "\n")
        fh.flush()

    return log


def save_json(obj, path):
    pathlib.Path(path).write_text(json.dumps(obj, indent=2, default=str))
