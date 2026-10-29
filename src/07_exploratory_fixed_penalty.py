"""POST-UNBLINDING EXPLORATORY analysis (not in the frozen protocol).

The frozen inner-CV tuning chose (lam1, lam2) = (inf, inf) for MB and MBT in every fold, so both
collapsed to contrast ATEs. This script forces finite penalties (no inf in the grid) to show how
much held-out tau-risk the moderator models lose or gain when they are made to use the features.

Output: reports/results_tables/exploratory_fixed_penalty.csv
"""
import importlib

import pandas as pd

from common import SEED
import data_access as D
import estimation as E

A = importlib.import_module("05_primary_analysis")
SETTINGS = [(3e4, float("inf")), (3e3, float("inf")), (3e2, float("inf")), (3e1, float("inf")), (3e3, 3e2)]
PAIRS = {"M0_vs_MB": ("M0", "MB", "risk"), "MB_vs_MBT": ("MB", "MBT", "risk"),
         "M0_vs_MS": ("M0", "MS", "risk"), "MS_vs_MST": ("MS", "MST", "risk"),
         "policy_MBT_minus_MB": ("MB", "MBT", "policy")}


def main():
    A.check_protocol()
    df = A.join_outcomes(D.load_frame("b24_k20"), A.load_outcomes())
    seeds = [SEED + 101 * r for r in range(2)]
    rows = []
    for lam1, lam2 in SETTINGS:
        E.LAM1_GRID, E.LAM2_GRID = [lam1], [lam2]
        outs = A.run(df, seeds, f"fixed_{lam1:g}_{lam2:g}")
        res = A.summarize_runs(outs, df, PAIRS)
        for name, r in res.items():
            rows.append({"lam1": lam1, "lam2": lam2, "comparison": name,
                         **{k: r[k] for k in ("estimate", "se", "ci_low", "ci_high", "p_one_sided")}})
        A.log(f"fixed penalty {lam1:g}/{lam2:g} done")
    pd.DataFrame(rows).to_csv(A.OUT / "exploratory_fixed_penalty.csv", index=False)


if __name__ == "__main__":
    main()
