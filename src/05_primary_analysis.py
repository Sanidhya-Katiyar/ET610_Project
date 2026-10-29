"""Phase C — confirmatory and pre-specified robustness analyses (run ONLY after protocol freeze).

Refuses to run unless protocol/ANALYSIS_PROTOCOL.md matches the SHA-256 in protocol/PROTOCOL_HASH.txt.
Writes the first outcome-unblinding timestamp to logs/UNBLINDING.txt.

Outputs: reports/results_tables/*.csv, reports/results_tables/results.json
Usage:   python src/05_primary_analysis.py [--quick]
"""
import argparse
import datetime as dt
import hashlib
import json

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.ensemble import HistGradientBoostingRegressor

from common import AUDIT, DERIVED, LOGS, REPORTS, ROOT, SEED, connect, create_raw_views, make_logger, save_json
import data_access as D
import estimation as E
import pipeline as P

OUT = REPORTS / "results_tables"
OUT.mkdir(exist_ok=True)
log = make_logger("05_primary_analysis")


# ------------------------------------------------------------------------------------ guards
def check_protocol():
    proto = ROOT / "protocol" / "ANALYSIS_PROTOCOL.md"
    hfile = ROOT / "protocol" / "PROTOCOL_HASH.txt"
    if not proto.exists() or not hfile.exists():
        raise SystemExit("Protocol or hash file missing: refusing to touch outcomes.")
    h = hashlib.sha256(proto.read_bytes()).hexdigest()
    expected = hfile.read_text().split()[0]
    if h != expected:
        raise SystemExit(f"Protocol hash mismatch ({h} != {expected}): protocol changed after freeze.")
    return h


def join_outcomes(rows, outcomes):
    """Many-to-one join of unit-contrast rows to unit outcomes; fails loudly on duplicates or gaps."""
    assert not outcomes.unit_id.duplicated().any(), "duplicate unit_id in outcomes"
    j = rows.merge(outcomes[["unit_id", "assignment_completed"]], on="unit_id", how="left", validate="many_to_one")
    assert j.assignment_completed.notna().all(), "missing outcome for some randomized unit"
    return j.rename(columns={"assignment_completed": "y"})


def load_outcomes():
    con = connect(read_only=False)
    create_raw_views(con)
    o = con.execute("SELECT assignment_log_id AS unit_id, assignment_completed, problems_completed FROM raw_exp_targets").df()
    return o


# ------------------------------------------------------------------------------------ helpers
def summarize_runs(outs, df, comparisons, weights="student"):
    """Average point estimates and SEs over cross-fitting repetitions."""
    res = {}
    for name, (a, b, kind) in comparisons.items():
        ests, ses = [], []
        for out in outs:
            fn = P.risk_diff if kind == "risk" else P.policy_gain
            e, s = fn(out, df, a, b, weights=weights)
            ests.append(e)
            ses.append(s)
        est, se = float(np.mean(ests)), float(np.mean(ses))
        # se == 0 happens when both models give identical held-out predictions in every fold (e.g. both
        # tuned to the constant model): the loss difference is identically 0 and H0 cannot be rejected.
        degenerate = se == 0.0
        z = float("nan") if degenerate else est / se
        res[name] = {"estimate": est, "se": se, "ci_low": est - 1.96 * se, "ci_high": est + 1.96 * se,
                     "z": z, "p_one_sided": 1.0 if degenerate else float(1 - stats.norm.cdf(z)),
                     "degenerate_identical_predictions": bool(degenerate),
                     "rep_estimates": [float(x) for x in ests]}
    return res


def run(df, seeds, label, **kw):
    outs = []
    for r, sd in enumerate(seeds):
        out, lam = P.crossfit(df, D.XB, D.XT, seed=sd, **kw)
        outs.append(out)
        lam.assign(rep=r).to_csv(OUT / f"lambdas_{label}_rep{r}.csv", index=False)
        log(f"{label}: rep {r} done")
    return outs


COMPARISONS = {
    "H1_primary_MB_vs_MBT": ("MB", "MBT", "risk"),
    "S1_MS_vs_MST": ("MS", "MST", "risk"),
    "S2_policy_MBT_minus_MB": ("MB", "MBT", "policy"),
    "S3_M0_vs_MB": ("M0", "MB", "risk"),
    "D_M0_vs_MBT": ("M0", "MBT", "risk"),
    "D_M0_vs_MS": ("M0", "MS", "risk"),
    "D_MS_vs_MBT": ("MS", "MBT", "risk"),
    "D_policy_MB_minus_fixed": ("fixed", "MB", "policy"),
    "D_policy_MBT_minus_fixed": ("fixed", "MBT", "policy"),
}
PRIMARY_ONLY = {k: v for k, v in COMPARISONS.items() if k in ("H1_primary_MB_vs_MBT",)}


def holm(pvals):
    names = sorted(pvals, key=pvals.get)
    m, adj, running = len(names), {}, 0.0
    for i, n in enumerate(names):
        running = max(running, min(1.0, (m - i) * pvals[n]))
        adj[n] = running
    return adj


class FlexCATE:
    """Flexible-learner sensitivity: tau(x) = alpha_c + g(x, experiment), g = gradient boosting on the
    contrast-demeaned training pseudo-outcome (experiment code as a categorical feature)."""

    def __init__(self, exp_codes_full):
        self.exp_codes_full = exp_codes_full

    def fit(self, X, phi, codes, n_codes, groups, seed):
        self.alpha, cnt = E.group_means(phi, codes, n_codes)
        self.alpha = np.where(cnt > 0, self.alpha, phi.mean())
        ex = self.exp_codes_full[codes][:, None]
        self.g = HistGradientBoostingRegressor(max_iter=200, learning_rate=0.03, max_leaf_nodes=15,
                                               min_samples_leaf=500, l2_regularization=10.0,
                                               categorical_features=[X.shape[1]], random_state=seed)
        self.g.fit(np.c_[X, ex], phi - self.alpha[codes])
        return self

    def predict(self, X, codes):
        return self.alpha[codes] + self.g.predict(np.c_[X, self.exp_codes_full[codes][:, None]])


def contrast_itt(d, ycol="y"):
    g = d.groupby("contrast_id")
    rows = []
    for cid, x in g:
        y1, y0 = x.loc[x.z == 1, ycol], x.loc[x.z == 0, ycol]
        est = y1.mean() - y0.mean()
        se = np.sqrt(y1.var(ddof=1) / len(y1) + y0.var(ddof=1) / len(y0))
        rows.append({"contrast_id": cid, "n0": len(y0), "n1": len(y1), "ate": est, "se": se})
    return pd.DataFrame(rows)


def dersimonian_laird(est, se):
    w = 1 / se ** 2
    mu = np.sum(w * est) / w.sum()
    Q = np.sum(w * (est - mu) ** 2)
    k = len(est)
    tau2 = max(0.0, (Q - (k - 1)) / (w.sum() - (w ** 2).sum() / w.sum()))
    return {"k": int(k), "Q": float(Q), "Q_p": float(1 - stats.chi2.cdf(Q, k - 1)), "tau2": float(tau2),
            "tau_sd_across_contrasts": float(np.sqrt(tau2)), "I2": float(max(0, (Q - (k - 1)) / Q)) if Q > 0 else 0.0}


def balance(d):
    rows = []
    for cid, x in d.groupby("contrast_id"):
        for c in D.XB + D.XT + ["s"]:
            a, b = x.loc[x.z == 1, c], x.loc[x.z == 0, c]
            sd = np.sqrt((a.var() + b.var()) / 2)
            rows.append({"contrast_id": cid, "feature": c, "smd": (a.mean() - b.mean()) / sd if sd > 0 else 0.0})
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="1 repetition everywhere (debugging only)")
    args = ap.parse_args()
    phash = check_protocol()
    R = 1 if args.quick else 5
    R2 = 1 if args.quick else 2
    seeds = [SEED + 101 * r for r in range(R)]
    results = {"protocol_sha256": phash, "started": dt.datetime.now().isoformat(timespec="seconds")}

    outcomes = load_outcomes()
    df = join_outcomes(D.load_frame("b24_k20"), outcomes)
    stamp = dt.datetime.now().isoformat(timespec="seconds")
    with open(LOGS / "UNBLINDING.txt", "a") as fh:
        fh.write(f"{stamp} first join of experimental outcomes to treatment arms (protocol sha256 {phash})\n")
    log(f"UNBLINDED at {stamp}")

    # ---- Primary analysis ---------------------------------------------------------------
    results["sample"] = {"rows": int(len(df)), "contrasts": int(df.contrast_id.nunique()),
                         "experiments": int(df.experiment_id.nunique()), "units": int(df.unit_id.nunique()),
                         "students": int(df.student_id.nunique()), "groups": int(df.group.nunique())}
    outs = run(df, seeds, "primary")
    prim = summarize_runs(outs, df, COMPARISONS)
    pv = {k: prim[k]["p_one_sided"] for k in ("S1_MS_vs_MST", "S2_policy_MBT_minus_MB", "S3_M0_vs_MB")}
    for k, v in holm(pv).items():
        prim[k]["p_holm"] = v
    results["primary"] = prim
    results["primary_equal_contrast_weights"] = summarize_runs(outs, df, COMPARISONS, weights="contrast")
    # Model risks relative to the constant model, for interpretation (units: squared probability).
    # BLP-type calibration diagnostic of the trajectory increment.
    blp = []
    for out in outs:
        codes, nc = E.contrast_index(df.contrast_id.to_numpy())
        b, s = E.blp_increment(out.phi.to_numpy(), out.tau_MB.to_numpy(), out.tau_MBT.to_numpy(), codes, nc,
                               df.group.to_numpy())
        blp.append((b, s))
    results["blp_increment"] = {"coef_tauB": float(np.mean([x[0][0] for x in blp])),
                                "se_tauB": float(np.mean([x[1][0] for x in blp])),
                                "coef_increment": float(np.mean([x[0][1] for x in blp])),
                                "se_increment": float(np.mean([x[1][1] for x in blp]))}
    # Spread of predicted CATEs (how much heterogeneity each model claims).
    results["predicted_cate_sd_within_contrast"] = {
        m: float(np.mean([out.groupby(df.contrast_id.values)[f"tau_{m}"].std().mean() for out in outs]))
        for m in P.MODELS}
    # SESOI calibrated test against the outcome-blind simulation (scenario d).
    sim = pd.read_csv(AUDIT / "simulation_reps.csv")
    dsim = sim.loc[sim.scenario == "d_traj_sesoi", "primary_est"]
    obs = prim["H1_primary_MB_vs_MBT"]["estimate"]
    results["sesoi_test"] = {"observed": obs, "sim_reps": int(len(dsim)),
                             "p_sesoi_lower_tail": float((np.sum(dsim <= obs) + 1) / (len(dsim) + 1)),
                             "sim_d_mean": float(dsim.mean()), "sim_d_q05": float(dsim.quantile(0.05))}
    # Protocol §6/§10: H1 additionally requires the estimate to exceed the 97.5th percentile of the
    # pooled simulated null distribution (scenarios a and b).
    nsim = sim.loc[sim.scenario.isin(["a_null", "b_level"]), "primary_est"]
    h1 = prim["H1_primary_MB_vs_MBT"]
    results["h1_decision"] = {"ci_low_gt0": bool(h1["ci_low"] > 0), "null_q975": float(nsim.quantile(0.975)),
                              "exceeds_null_q975": bool(obs > nsim.quantile(0.975)),
                              "p_null_upper_tail": float((np.sum(nsim >= obs) + 1) / (len(nsim) + 1))}
    results["h1_decision"]["supported"] = (results["h1_decision"]["ci_low_gt0"]
                                           and results["h1_decision"]["exceeds_null_q975"])
    log("primary done")

    # ---- Robustness --------------------------------------------------------------------
    rob = {}
    seeds2 = seeds[:R2]
    for variant in ("b24_k10", "b0_k20"):
        d = join_outcomes(D.load_frame(variant), outcomes)
        rob[variant] = {"rows": int(len(d)), "contrasts": int(d.contrast_id.nunique()),
                        **summarize_runs(run(d, seeds2, variant), d, PRIMARY_ONLY)}
    rob["p_half"] = summarize_runs(run(df, seeds2, "p_half", p_mode="half"), df, PRIMARY_ONLY)
    rob["p_all"] = summarize_runs(run(df, seeds2[:1], "p_all", p_mode="all"), df, PRIMARY_ONLY)
    # One contrast per experiment: the largest primary contrast (outcome-blind choice by size).
    big = df.groupby(["experiment_id", "contrast_id"]).size().reset_index(name="n").sort_values("n").groupby("experiment_id").tail(1)
    d1 = df[df.contrast_id.isin(big.contrast_id)].reset_index(drop=True)
    rob["one_contrast_per_experiment"] = {"rows": int(len(d1)), "contrasts": int(d1.contrast_id.nunique()),
                                          **summarize_runs(run(d1, seeds2, "one_per_exp"), d1, PRIMARY_ONLY)}
    # Held-out experiments (transfer to unseen contrasts in the corpus).
    rob["experiment_heldout"] = summarize_runs(run(df, seeds2[:1], "exp_heldout", group_col="experiment_id", p_mode="all"),
                                               df, PRIMARY_ONLY)
    # Stable-allocation subset.
    alloc = pd.read_csv(AUDIT / "allocation_over_time.csv")
    stable = set(alloc.loc[~(alloc.p_time_stability < 0.001), "contrast_id"])
    ds = df[df.contrast_id.isin(stable)].reset_index(drop=True)
    rob["stable_allocation_contrasts"] = {"contrasts": int(ds.contrast_id.nunique()),
                                          **summarize_runs(run(ds, seeds2[:1], "stable"), ds, PRIMARY_ONLY)}
    # Flexible learner.
    exp_codes = pd.factorize(df.experiment_id)[0]
    cid_codes, _ = E.contrast_index(df.contrast_id.to_numpy())
    exp_of_contrast = np.zeros(cid_codes.max() + 1, dtype=int)
    exp_of_contrast[cid_codes] = exp_codes
    learner = lambda X, phi, codes, nc, groups, seed: FlexCATE(exp_of_contrast).fit(X, phi, codes, nc, groups, seed)
    rob["flexible_gbm"] = summarize_runs(run(df, seeds2[:1], "flex", tau_learner=learner, models=("M0", "MB", "MBT")),
                                         df, {"H1_flex_MB_vs_MBT": ("MB", "MBT", "risk"), "flex_M0_vs_MB": ("M0", "MB", "risk")})
    # Permutation placebo: Z permuted within contrast; true Delta_R for any model is <= 0.
    placebo = []
    rng = np.random.default_rng(SEED + 999)
    for k in range(R):
        dp = df.copy()
        dp["z"] = dp.groupby("contrast_id").z.transform(lambda s: rng.permutation(s.to_numpy()))
        o, _ = P.crossfit(dp, D.XB, D.XT, seed=SEED + 500 + k, models=("M0", "MB", "MBT"))
        e, s = P.risk_diff(o, dp, "MB", "MBT")
        e0, s0 = P.risk_diff(o, dp, "M0", "MB")
        placebo.append({"draw": k, "MB_vs_MBT": e, "se": s, "M0_vs_MB": e0, "se0": s0})
        log(f"placebo {k} done")
    rob["permutation_placebo"] = placebo
    results["robustness"] = rob

    # ---- ITT, heterogeneity of ATEs, population check, balance ------------------------------
    itt_prim = contrast_itt(df)
    dall = join_outcomes(D.load_frame("b24_k20", all_randomized=True), outcomes)
    itt_all = contrast_itt(dall)
    itt = itt_prim.merge(itt_all, on="contrast_id", suffixes=("_hist", "_all"))
    itt.to_csv(OUT / "contrast_itt.csv", index=False)
    results["itt"] = {
        "primary_sample": dersimonian_laird(itt_prim.ate.to_numpy(), itt_prim.se.to_numpy()),
        "all_randomized": dersimonian_laird(itt_all.ate.to_numpy(), itt_all.se.to_numpy()),
        "share_contrasts_ci_excludes_0_hist": float((np.abs(itt_prim.ate / itt_prim.se) > 1.96).mean()),
        "corr_ate_hist_vs_all": float(np.corrcoef(itt.ate_hist, itt.ate_all)[0, 1]),
        "mean_abs_ate_hist": float(itt.ate_hist.abs().mean()), "mean_abs_ate_all": float(itt.ate_all.abs().mean()),
        "all_randomized_rows": int(len(dall)), "completion_rate_primary": float(df.drop_duplicates("unit_id").y.mean()),
        "completion_rate_all": float(dall.drop_duplicates("unit_id").y.mean()),
    }
    bal = balance(df)
    bal.to_csv(OUT / "balance_smd.csv", index=False)
    results["balance"] = {"share_abs_smd_gt_0.1": float((bal.smd.abs() > 0.1).mean()),
                          "share_abs_smd_gt_0.2": float((bal.smd.abs() > 0.2).mean()),
                          "median_abs_smd": float(bal.smd.abs().median())}
    # Exploratory: problems_completed per contrast (all randomized; meaning differs by completion status).
    dpc = dall.drop(columns="y").merge(outcomes[["unit_id", "problems_completed"]], on="unit_id")
    pc = contrast_itt(dpc, "problems_completed")
    pc.to_csv(OUT / "exploratory_problems_completed_itt.csv", index=False)
    results["exploratory_problems_completed"] = dersimonian_laird(pc.ate.to_numpy(), pc.se.to_numpy())
    # Exploratory: best linear projection of tau on standardized features (shared across contrasts,
    # within-contrast centred), using the held-out pseudo-outcomes of repetition 0 (E[phi|X] = tau(X)).
    # Restricted to Control-vs-Treatment contrasts, where the orientation (z = 1: Treatment-labelled arm)
    # is meaningful; Treatment-vs-Treatment orientations are arbitrary and would cancel.
    feats = D.XB + D.XT
    ct = (df.arm0.str.startswith("Control") & df.arm1.str.startswith("Treatment")).to_numpy()
    dct = df[ct]
    Xs = dct[feats].fillna(dct[feats].median())
    Xs = ((Xs - Xs.mean()) / Xs.std()).to_numpy()
    codes, nc = E.contrast_index(dct.contrast_id.to_numpy())
    Xd = Xs - np.vstack([E.group_means(Xs[:, j], codes, nc)[0][codes] for j in range(Xs.shape[1])]).T
    phi = outs[0].phi.to_numpy()[ct]
    yd = phi - E.group_means(phi, codes, nc)[0][codes]
    A = Xd.T @ Xd
    beta = np.linalg.solve(A, Xd.T @ yd)
    e = yd - Xd @ beta
    gcodes, G = E.contrast_index(dct.group.to_numpy())
    S = np.vstack([np.bincount(gcodes, weights=e * Xd[:, j], minlength=G) for j in range(Xd.shape[1])]).T
    Ai = np.linalg.inv(A)
    se = np.sqrt(np.diag(G / (G - 1) * Ai @ (S.T @ S) @ Ai))
    blp_feat = pd.DataFrame({"feature": feats, "coef_pp_per_sd": 100 * beta, "se_pp": 100 * se})
    blp_feat["z"] = blp_feat.coef_pp_per_sd / blp_feat.se_pp
    blp_feat["p_two_sided_unadjusted"] = 2 * (1 - stats.norm.cdf(blp_feat.z.abs()))
    blp_feat["n_rows"], blp_feat["n_contrasts"] = int(ct.sum()), int(dct.contrast_id.nunique())
    blp_feat.to_csv(OUT / "exploratory_blp_by_feature.csv", index=False)
    results["finished"] = dt.datetime.now().isoformat(timespec="seconds")
    save_json(results, OUT / "results.json")
    log("analysis complete")


if __name__ == "__main__":
    main()
