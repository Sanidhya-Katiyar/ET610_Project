"""Figures from generated result tables (no hand-copied numbers). Writes figures/*.png."""
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from common import AUDIT, FIGURES, REPORTS  # noqa: E402

RES = REPORTS / "results_tables"
plt.rcParams.update({"figure.dpi": 150, "font.size": 9, "axes.spines.top": False, "axes.spines.right": False})
SCALE = 1e4  # squared-probability units x 1e4 (= squared percentage points / 1)


def forest(ax, labels, est, lo, hi, title, xlabel):
    y = np.arange(len(labels))[::-1]
    ax.errorbar(est, y, xerr=[np.array(est) - np.array(lo), np.array(hi) - np.array(est)], fmt="o", color="#1f4e79",
                ecolor="#7f9fbf", capsize=2, ms=4)
    ax.axvline(0, color="grey", lw=0.8, ls="--")
    ax.set_yticks(y, labels)
    ax.set_title(title, loc="left")
    ax.set_xlabel(xlabel)


def main():
    r = json.loads((RES / "results.json").read_text())
    p, rob = r["primary"], r["robustness"]

    # Figure 1: primary and robustness Delta_R(MB -> MBT).
    rows = [("Primary (student-contrast weights)", p["H1_primary_MB_vs_MBT"]),
            ("Equal-contrast weights", r["primary_equal_contrast_weights"]["H1_primary_MB_vs_MBT"]),
            ("K = 10 window, >= 5 prior", rob["b24_k10"]["H1_primary_MB_vs_MBT"]),
            ("Strict 60 s cutoff", rob["b0_k20"]["H1_primary_MB_vs_MBT"]),
            ("p = 1/2 fixed", rob["p_half"]["H1_primary_MB_vs_MBT"]),
            ("p = full-contrast share", rob["p_all"]["H1_primary_MB_vs_MBT"]),
            ("One contrast per experiment", rob["one_contrast_per_experiment"]["H1_primary_MB_vs_MBT"]),
            ("Held-out experiments", rob["experiment_heldout"]["H1_primary_MB_vs_MBT"]),
            ("Stable-allocation contrasts", rob["stable_allocation_contrasts"]["H1_primary_MB_vs_MBT"]),
            ("Flexible GBM learner", rob["flexible_gbm"]["H1_flex_MB_vs_MBT"])]
    for k, pl in enumerate(rob["permutation_placebo"]):
        rows.append((f"Permutation placebo {k + 1}", {"estimate": pl["MB_vs_MBT"], "ci_low": pl["MB_vs_MBT"] - 1.96 * pl["se"],
                                                      "ci_high": pl["MB_vs_MBT"] + 1.96 * pl["se"]}))
    fig, ax = plt.subplots(figsize=(6.5, 4.6))
    forest(ax, [x[0] for x in rows], [x[1]["estimate"] * SCALE for x in rows], [x[1]["ci_low"] * SCALE for x in rows],
           [x[1]["ci_high"] * SCALE for x in rows],
           "Delta_R(MB -> MBT): held-out tau-risk reduction",
           "Delta_R x 1e4 (positive = trajectories help); 95% CI")
    fig.tight_layout()
    fig.savefig(FIGURES / "fig1_delta_r_primary_robustness.png")

    # Figure 2: risk reduction relative to the constant model, and policy gains.
    fig, axes = plt.subplots(1, 2, figsize=(8, 2.8))
    comps = [("MB vs M0", p["S3_M0_vs_MB"]), ("MBT vs M0", p["D_M0_vs_MBT"]), ("MS vs M0", p["D_M0_vs_MS"]),
             ("MBT vs MS", p["D_MS_vs_MBT"]), ("MST vs MS", p["S1_MS_vs_MST"])]
    forest(axes[0], [c[0] for c in comps], [c[1]["estimate"] * SCALE for c in comps],
           [c[1]["ci_low"] * SCALE for c in comps], [c[1]["ci_high"] * SCALE for c in comps],
           "tau-risk reduction (x 1e4)", "positive = second model better")
    pol = [("MB vs best fixed arm", p["D_policy_MB_minus_fixed"]), ("MBT vs best fixed arm", p["D_policy_MBT_minus_fixed"]),
           ("MBT vs MB", p["S2_policy_MBT_minus_MB"])]
    forest(axes[1], [c[0] for c in pol], [100 * c[1]["estimate"] for c in pol], [100 * c[1]["ci_low"] for c in pol],
           [100 * c[1]["ci_high"] for c in pol], "Held-out policy value gain", "completion, percentage points")
    fig.tight_layout()
    fig.savefig(FIGURES / "fig2_model_risks_and_policy.png")

    # Figure 3: simulation power and the SESOI reference distribution.
    sim = pd.read_csv(AUDIT / "simulation_summary.csv").set_index("scenario")
    reps = pd.read_csv(AUDIT / "simulation_reps.csv")
    fig, axes = plt.subplots(1, 2, figsize=(8, 2.8))
    order = ["a_null", "b_level", "c_traj_small", "d_traj_sesoi", "e_traj_mixed", "f_traj_large"]
    rr = sim.loc[order, "primary_reject_lower_gt0"]
    mc = sim.loc[order, "primary_reject_mcse"]
    axes[0].bar(range(len(order)), rr, yerr=1.96 * mc, color="#7f9fbf", capsize=2)
    axes[0].axhline(0.025, color="red", lw=0.8, ls="--", label="nominal 2.5%")
    axes[0].set_xticks(range(len(order)), [o.replace("_", "\n", 1) for o in order], fontsize=7)
    axes[0].set_ylim(0, 1)
    axes[0].set_ylabel("P(95% CI lower bound > 0)")
    axes[0].set_title("Simulated H1 rejection rate", loc="left")
    axes[0].legend(frameon=False, fontsize=7)
    dd = reps.loc[reps.scenario == "d_traj_sesoi", "primary_est"] * SCALE
    da = reps.loc[reps.scenario == "a_null", "primary_est"] * SCALE
    axes[1].hist(da, bins=25, alpha=0.6, label="null (a)", color="grey")
    axes[1].hist(dd, bins=25, alpha=0.6, label="SESOI (d)", color="#1f4e79")
    axes[1].axvline(p["H1_primary_MB_vs_MBT"]["estimate"] * SCALE, color="red", lw=1.2, label="observed")
    axes[1].set_xlabel("Delta_R x 1e4")
    axes[1].set_title("Simulated vs observed Delta_R", loc="left")
    axes[1].legend(frameon=False, fontsize=7)
    fig.tight_layout()
    fig.savefig(FIGURES / "fig3_simulation_power_sesoi.png")

    # Figure 4: contrast-level ITT estimates (primary sample) and population check.
    itt = pd.read_csv(RES / "contrast_itt.csv")
    fig, axes = plt.subplots(1, 2, figsize=(8, 2.8))
    o = itt.sort_values("ate_hist")
    axes[0].errorbar(range(len(o)), 100 * o.ate_hist, yerr=196 * o.se_hist, fmt=".", ms=2, elinewidth=0.4, color="#1f4e79")
    axes[0].axhline(0, color="grey", lw=0.8)
    axes[0].set_xlabel("contrast (sorted by estimate)")
    axes[0].set_ylabel("ATE on completion (pp), 95% CI")
    axes[0].set_title("Contrast-level ITT effects, primary sample", loc="left")
    axes[1].scatter(100 * itt.ate_all, 100 * itt.ate_hist, s=5, color="#1f4e79")
    lim = [100 * min(itt.ate_all.min(), itt.ate_hist.min()), 100 * max(itt.ate_all.max(), itt.ate_hist.max())]
    axes[1].plot(lim, lim, color="grey", lw=0.8, ls="--")
    axes[1].set_xlabel("ATE, all randomized (pp)")
    axes[1].set_ylabel("ATE, >= 10-history sample (pp)")
    axes[1].set_title("Population check", loc="left")
    fig.tight_layout()
    fig.savefig(FIGURES / "fig4_contrast_itt.png")

    # Figure 5: exploratory per-feature linear projection of tau (Control-vs-Treatment contrasts).
    b = pd.read_csv(RES / "exploratory_blp_by_feature.csv")
    fig, ax = plt.subplots(figsize=(6, 5))
    forest(ax, list(b.feature), list(b.coef_pp_per_sd), list(b.coef_pp_per_sd - 1.96 * b.se_pp),
           list(b.coef_pp_per_sd + 1.96 * b.se_pp),
           "EXPLORATORY: CATE projection on features",
           "pp change in completion effect per SD (unadjusted 95% CI)")
    fig.tight_layout()
    fig.savefig(FIGURES / "fig5_exploratory_feature_projection.png")
    print("figures written")


if __name__ == "__main__":
    main()
