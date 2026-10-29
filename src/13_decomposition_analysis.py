"""Decomposition analysis (protocol/DECOMPOSITION_PROTOCOL.md). Run once after the protocol freeze.

Output: reports/results_tables/decomposition_results.json
"""
import datetime as dt
import hashlib
import importlib

import numpy as np
import pandas as pd

from common import LOGS, REPORTS, ROOT, SEED, make_logger, save_json
import data_access as D
import decomposition as Dc
import pipeline as P

A = importlib.import_module("05_primary_analysis")
S = importlib.import_module("12_decomposition_simulation")
log = make_logger("13_decomposition_analysis")
OUT = REPORTS / "results_tables"
SEEDS = [SEED + 101 * r for r in range(5)]
BETWEEN_SD_ALL = 0.0355  # DerSimonian-Laird between-contrast SD, all randomized units (results.json -> itt)


def check_hash():
    proto = ROOT / "protocol" / "DECOMPOSITION_PROTOCOL.md"
    expected = (ROOT / "protocol" / "DECOMPOSITION_PROTOCOL_HASH.txt").read_text().split()[0]
    h = hashlib.sha256(proto.read_bytes()).hexdigest()
    if h != expected:
        raise SystemExit("decomposition protocol changed after freeze")
    return h


def run(df, label):
    keys = (df.contrast_id.to_numpy(), df.district.to_numpy(), df.teacher.to_numpy(), df.class_id.to_numpy())
    ests, ses = [], []
    for sd in SEEDS:
        out, _ = P.crossfit(df, D.XB, D.XT, seed=sd, models=("M0",), group_col="ctx_group")
        e, s = Dc.decompose(*keys, out.phi.to_numpy())
        ests.append(e)
        ses.append(s)
        log(f"{label}: seed {sd} done")
    res = {"rows": int(len(df)), "contrasts": int(df.contrast_id.nunique()), "districts": int(df.district.nunique()),
           "teachers": int(df.teacher.nunique()), "classes": int(df.class_id.nunique())}
    for k in ests[0]:
        v, se = float(np.mean([e[k] for e in ests])), float(np.mean([s[k] for s in ses]))
        res[k] = {"V": v, "se": se, "ci_low": v - 1.96 * se, "ci_high": v + 1.96 * se,
                  "sd_pp": 100 * float(np.sqrt(max(v, 0))), "sd_upper_pp": 100 * float(np.sqrt(max(v + 1.96 * se, 0))),
                  "seed_estimates": [float(e[k]) for e in ests]}
    return res


def main():
    h = check_hash()
    outcomes = A.load_outcomes()
    stamp = dt.datetime.now().isoformat(timespec="seconds")
    with open(LOGS / "UNBLINDING.txt", "a") as fh:
        fh.write(f"{stamp} decomposition analysis: outcomes joined by class/teacher/district (protocol sha256 {h})\n")
    df = A.join_outcomes(S.build_frame(all_randomized=True), outcomes)
    results = {"protocol_sha256": h, "started": stamp}
    prim = run(df, "all_randomized")
    t = prim["total_context"]
    detected = t["ci_low"] > 0
    if detected:
        interp = ("context moderation is at least comparable to differences between interventions"
                  if t["sd_pp"] >= 100 * BETWEEN_SD_ALL else
                  "context moderation is present but smaller than intervention differences")
    else:
        interp = f"context moderation is below {t['sd_upper_pp']:.1f} pp" + (
            "; cannot rule out context variation of the size of intervention differences"
            if t["sd_upper_pp"] >= 100 * BETWEEN_SD_ALL else "")
    results["primary"] = prim
    results["decision"] = {"detected": bool(detected), "between_contrast_sd_pp": 100 * BETWEEN_SD_ALL, "interpretation": interp}
    # Secondary sensitivity analyses.
    df_hist = df[df.n_prior_stats >= 10].reset_index(drop=True)
    results["sensitivity_primary_history_sample"] = run(df_hist, "history_sample")
    df_real = df[~df.district.astype(str).str.startswith("T")].reset_index(drop=True)
    results["sensitivity_real_district_units"] = run(df_real, "real_district_units")
    results["finished"] = dt.datetime.now().isoformat(timespec="seconds")
    save_json(results, OUT / "decomposition_results.json")
    log(f"decision: {results['decision']}")
    print(results["decision"])


if __name__ == "__main__":
    main()
