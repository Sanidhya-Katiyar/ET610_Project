"""POST-UNBLINDING EXPLORATORY analysis (not in the frozen protocol): prognosis vs. benefit.

For a prognostic score, a level feature and two trajectory features, compares by within-sample quintile:
  (a) completion rate (prognosis), and
  (b) the held-out oriented effect: DR pseudo-outcome phi times sign of the training-fold contrast ATE,
      i.e. the effect of receiving the contrast's (in-fold) better arm. Flat (b) = no detectable
      effect modification along that axis.
Also reports the AUC of the remnant score for completion in the experimental sample.

Output: reports/results_tables/exploratory_prognosis_vs_benefit.csv, .json
"""
import importlib

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from common import SEED, save_json
import data_access as D
import estimation as E
import pipeline as P

A = importlib.import_module("05_primary_analysis")
AXES = {"s": "Remnant prognostic score", "b_correct_all": "Lifetime correctness (level)",
        "t_slope_correct_norm": "Correctness slope (trajectory)", "t_recent_shift_correct": "Recent shift (trajectory)"}


def main():
    A.check_protocol()
    df = A.join_outcomes(D.load_frame("b24_k20"), A.load_outcomes())
    out, _ = P.crossfit(df, D.XB, D.XT, seed=SEED, models=("M0",))
    benefit = out.phi.to_numpy() * np.sign(out.tau_M0.to_numpy())
    groups = df.group.to_numpy()
    rows, tests = [], {}
    for col, label in AXES.items():
        ok = df[col].notna().to_numpy()
        q = pd.qcut(df.loc[ok, col].rank(method="first"), 5, labels=False).to_numpy()
        for k in range(5):
            m = np.zeros(len(df), bool)
            m[np.flatnonzero(ok)[q == k]] = True
            est, se = E.cluster_mean_se(benefit[m], groups[m], np.ones(m.sum()))
            yr, yse = E.cluster_mean_se(df.y.to_numpy(float)[m], groups[m], np.ones(m.sum()))
            rows.append({"axis": col, "label": label, "quintile": k + 1, "n": int(m.sum()),
                         "completion": yr, "completion_se": yse, "benefit": est, "benefit_se": se})
        # Q5 - Q1 difference in oriented benefit, cluster SE over all rows.
        i1, i5 = np.flatnonzero(ok)[q == 0], np.flatnonzero(ok)[q == 4]
        d = np.zeros(len(df))
        d[i5] = benefit[i5] * len(df) / len(i5)
        d[i1] = -benefit[i1] * len(df) / len(i1)
        est, se = E.cluster_mean_se(d, groups, np.ones(len(df)))
        tests[col] = {"q5_minus_q1_benefit": est, "se": se, "z": est / se}
    auc = roc_auc_score(df.y, df.s)
    res = {"auc_remnant_score_experimental": float(auc), "mean_oriented_benefit": float(benefit.mean()),
           "q5_minus_q1": tests}
    pd.DataFrame(rows).to_csv(A.OUT / "exploratory_prognosis_vs_benefit.csv", index=False)
    save_json(res, A.OUT / "exploratory_prognosis_vs_benefit.json")
    A.log(f"prognosis vs benefit done: {res}")
    print(res)


if __name__ == "__main__":
    main()
