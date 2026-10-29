"""Option-B selection check (protocol/FOLLOWUP_SELECTION_CHECK.md, hash in FOLLOWUP_SELECTION_CHECK_HASH.txt).

Tests whether arm affects having a follow-up window. Computes no downstream outcome.
Output: reports/results_tables/followup_selection_check.json, followup_selection_by_contrast.csv
"""
import hashlib

import numpy as np
import pandas as pd
from scipy import stats

from common import DERIVED, REPORTS, ROOT, connect, create_raw_views, make_logger, save_json
import data_access as D

log = make_logger("11_followup_selection_check")
OUT = REPORTS / "results_tables"
DAY = 86400.0


def check_hash():
    proto = ROOT / "protocol" / "FOLLOWUP_SELECTION_CHECK.md"
    expected = (ROOT / "protocol" / "FOLLOWUP_SELECTION_CHECK_HASH.txt").read_text().split()[0]
    h = hashlib.sha256(proto.read_bytes()).hexdigest()
    if h != expected:
        raise SystemExit("selection-check protocol changed after freeze")
    return h


def dl_tau(d, se):
    w = 1 / se ** 2
    mu = np.sum(w * d) / np.sum(w)
    q = np.sum(w * (d - mu) ** 2)
    c = np.sum(w) - np.sum(w ** 2) / np.sum(w)
    return float(np.sqrt(max(0.0, (q - (len(d) - 1)) / c)))


def main():
    h = check_hash()
    con = connect(read_only=False)
    create_raw_views(con)
    con.execute(f"CREATE OR REPLACE TEMP VIEW units AS SELECT * FROM read_parquet('{(DERIVED / 'units_exp.parquet').as_posix()}')")
    w = con.execute(f"""WITH t AS (SELECT student_id, max(assignment_start_time) tmax FROM raw_exp_targets GROUP BY 1)
                        SELECT u.unit_id, u.assignment_start_time AS start, (t.tmax - u.assignment_start_time) / {DAY} AS window_days
                        FROM units u JOIN t USING (student_id)""").df()
    data_end = con.execute("SELECT max(assignment_start_time) FROM raw_exp_targets").fetchone()[0]
    df = D.load_frame("b24_k20")[["contrast_id", "z", "unit_id"]].merge(w, on="unit_id", how="left", validate="many_to_one")
    assert df.window_days.notna().all()

    res = {"protocol_sha256": h, "rows": int(len(df)), "contrasts": int(df.contrast_id.nunique())}
    tables = []
    for name, days in (("S7", 7), ("S30", 30), ("S_any", None)):
        if days is None:
            d = df.assign(s=(df.window_days > 1e-6).astype(float))
        else:
            d = df[df.start <= data_end - days * DAY].assign(s=lambda x: (x.window_days >= days).astype(float))
        g = d.groupby(["contrast_id", "z"]).s.agg(["mean", "count"]).unstack("z")
        g = g[(g["count"] >= 2).all(axis=1)]
        p0, p1, n0, n1 = g[("mean", 0)], g[("mean", 1)], g[("count", 0)], g[("count", 1)]
        dc = (p1 - p0).to_numpy()
        se = np.sqrt(p0 * (1 - p0) / n0 + p1 * (1 - p1) / n1).to_numpy()
        ok = se > 0
        dc, se = dc[ok], se[ok]
        q0 = float(np.sum((dc / se) ** 2))
        k = int(len(dc))
        tau = dl_tau(dc, se)
        exp_abs = float(np.mean(se * np.sqrt(2 / np.pi)))
        trim = (np.abs(p1 - p0) / np.maximum(p0, p1)).to_numpy()[ok]
        rej = stats.chi2.sf(q0, k) < 0.05
        r = {"rows": int(len(d)), "contrasts": k, "share_selected": float(d.s.mean()),
             "Q0": q0, "df": k, "p_Q0": float(stats.chi2.sf(q0, k)), "dl_tau": tau,
             "mean_abs_d": float(np.mean(np.abs(dc))), "expected_mean_abs_d_under_null": exp_abs,
             "share_contrasts_abs_z_gt_1.96": float(np.mean(np.abs(dc / se) > 1.96)),
             "median_lee_trim": float(np.median(trim)), "p90_lee_trim": float(np.quantile(trim, 0.9))}
        if not rej and tau <= 0.01:
            r["decision"] = "within-window analysis feasible"
        elif r["median_lee_trim"] <= 0.05:
            r["decision"] = "feasible only with bounds"
        else:
            r["decision"] = "not feasible"
        res[name] = r
        tables.append(pd.DataFrame({"indicator": name, "contrast_id": g.index[ok], "d": dc, "se": se}))
    pd.concat(tables).to_csv(OUT / "followup_selection_by_contrast.csv", index=False)
    save_json(res, OUT / "followup_selection_check.json")
    log(f"selection check: {res}")
    for k, v in res.items():
        print(k, v)


if __name__ == "__main__":
    main()
