"""Assemble the outcome-free analysis frame (one row per unit x contrast) for a given variant."""
import json

import pandas as pd

from common import AUDIT, DERIVED
import features as F

VARIANT_RULES = {
    # variant: (minimum prior assignments with statistics, contrast-eligibility column)
    "b24_k20": (10, "eligible_h10"),   # PRIMARY
    "b24_k10": (5, "eligible_h5"),     # pre-specified smaller-window sensitivity
    "b0_k20": (10, "eligible_h10_b0"),  # pre-specified strict-cutoff (60 s) sensitivity
}
XB = F.LEVEL_FEATURES
XT = F.TRAJECTORY_FEATURES


def load_frame(variant="b24_k20", all_randomized=False):
    """Outcome-free frame. all_randomized=True keeps every randomized unit of the variant's eligible
    contrasts regardless of history length (features may be missing) - used for the population ITT check."""
    min_hist, elig_col = VARIANT_RULES[variant]
    bounds = json.loads((AUDIT / "winsor_bounds.json").read_text())
    feats = pd.read_parquet(DERIVED / f"feat_exp_{variant}.parquet")
    for c, (lo, hi) in bounds.items():
        feats[c] = feats[c].clip(lo, hi)
    scores = pd.read_parquet(DERIVED / "scores_exp.parquet")
    scores = scores[scores.variant == variant][["unit_id", "s"]]
    units = pd.read_parquet(DERIVED / "units_exp.parquet")[["unit_id", "student_id", "class_id", "group", "experiment_id"]]
    con = pd.read_parquet(DERIVED / "contrasts.parquet")
    summ = pd.read_csv(AUDIT / "contrast_summary.csv")
    eligible = set(summ.loc[summ[elig_col], "contrast_id"])
    df = con[con.contrast_id.isin(eligible)].merge(units.drop(columns="experiment_id"), on="unit_id", how="left")
    df = df.merge(feats[["unit_id", "n_prior_stats"] + XB + XT], on="unit_id", how="left").merge(scores, on="unit_id", how="left")
    if not all_randomized:
        df = df[df.n_prior_stats >= min_hist]
    assert df.unit_id.notna().all() and df.group.notna().all()
    assert not df.duplicated(["unit_id", "contrast_id"]).any()
    return df.reset_index(drop=True)
