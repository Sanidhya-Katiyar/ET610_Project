"""Print-sized figures for the course report (report/figures/*.svg). Reads generated tables only."""
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from common import AUDIT, REPORTS, ROOT  # noqa: E402

RES = REPORTS / "results_tables"
OUT = ROOT / "report" / "figures"
OUT.mkdir(parents=True, exist_ok=True)
TEX_FIGS = ROOT / "report" / "overleaf" / "figures"
TEX_FIGS.mkdir(parents=True, exist_ok=True)
W = 6.85  # printed text width in inches (A4, 1.8 cm margins)
BLUE, ORANGE, AQUA, YELLOW = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
INK, INK2, GRID = "#0b0b0b", "#52514e", "#d9d8d4"
plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 10.5, "axes.titlesize": 10.5, "axes.labelsize": 10.5,
    "xtick.labelsize": 10, "ytick.labelsize": 10, "legend.fontsize": 10, "axes.spines.top": False,
    "axes.spines.right": False, "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2,
    "ytick.color": INK2, "axes.titlelocation": "left", "axes.titleweight": "bold", "svg.fonttype": "none", "pdf.fonttype": 42,
    "legend.frameon": False})


def save(fig, name):
    fig.savefig(OUT / f"{name}.svg", bbox_inches="tight")
    fig.savefig(OUT / f"{name}.png", dpi=200, bbox_inches="tight")
    fig.savefig(TEX_FIGS / f"{name}.pdf", bbox_inches="tight")  # vector copies for the Overleaf project
    plt.close(fig)


def zero_line(ax, vertical=True):
    (ax.axvline if vertical else ax.axhline)(0, color=INK2, lw=0.8, ls=(0, (3, 3)), zorder=0)


def fig_simulation():
    sim = pd.read_csv(AUDIT / "simulation_summary.csv").set_index("scenario")
    reps = pd.read_csv(AUDIT / "simulation_reps.csv")
    order = ["a_null", "b_level", "c_traj_small", "d_traj_sesoi", "e_traj_mixed", "f_traj_large"]
    names = ["Null", "Level\nonly", "Traj.\n1.5 pp", "Traj.\n3 pp", "Traj.\nmixed", "Traj.\n6 pp"]
    fig, ax = plt.subplots(1, 2, figsize=(W, 2.5), gridspec_kw={"width_ratios": [1.1, 1]})
    rr, mc = sim.loc[order, "primary_reject_lower_gt0"], sim.loc[order, "primary_reject_mcse"]
    ax[0].bar(range(6), rr, width=0.62, color=BLUE, yerr=1.96 * mc, ecolor=INK2, capsize=2)
    ax[0].axhline(0.025, color=ORANGE, lw=1.2, ls="--", label="nominal 2.5%")
    ax[0].legend(loc="upper left", bbox_to_anchor=(-0.02, 0.62), fontsize=10, handlelength=1.0, handletextpad=0.3)
    for i, (v, m) in enumerate(zip(rr, mc)):
        ax[0].text(i, v + 1.96 * m + 0.03, f"{v:.2f}", ha="center", fontsize=10, color=INK)
    ax[0].set_xticks(range(6), names, fontsize=10)
    ax[0].set_ylim(0, 1.12)
    ax[0].set_ylabel("P(95% CI excludes 0)")
    ax[0].set_title("(a) Rejection rate of H1")
    r = json.loads((RES / "results.json").read_text())
    s = 1e4
    a = reps.loc[reps.scenario == "a_null", "primary_est"] * s
    d = reps.loc[reps.scenario == "d_traj_sesoi", "primary_est"] * s
    bins = np.linspace(-3, 32, 71)
    ax[1].hist(a, bins=bins, color=INK2, alpha=0.55, label="Null")
    ax[1].hist(d, bins=bins, color=BLUE, alpha=0.8, label="SESOI (3 pp)")
    ax[1].axvline(r["primary"]["H1_primary_MB_vs_MBT"]["estimate"] * s, color=ORANGE, lw=1.8, label="Observed (0)")
    ax[1].axvline(r["sesoi_test"]["sim_d_q05"] * s, color=BLUE, lw=1, ls=":", label="SESOI 5th pct")
    ax[1].set_xlabel("ΔR(MB→MBT) × 10⁴")
    ax[1].set_ylabel("Replicates")
    ax[1].set_title("(b) Simulated vs observed ΔR")
    ax[1].legend(loc="upper center", bbox_to_anchor=(0.6, 1.0), fontsize=10)
    fig.tight_layout(w_pad=1.5)
    save(fig, "fig_simulation")


def fig_main():
    r = json.loads((RES / "results.json").read_text())
    p, rob = r["primary"], r["robustness"]
    rows = [("Primary", p["H1_primary_MB_vs_MBT"]),
            ("Equal-contrast weights", r["primary_equal_contrast_weights"]["H1_primary_MB_vs_MBT"]),
            ("K = 10 window", rob["b24_k10"]["H1_primary_MB_vs_MBT"]),
            ("60 s history cutoff", rob["b0_k20"]["H1_primary_MB_vs_MBT"]),
            ("p = 1/2 fixed", rob["p_half"]["H1_primary_MB_vs_MBT"]),
            ("p = full-sample share", rob["p_all"]["H1_primary_MB_vs_MBT"]),
            ("Stable-allocation only", rob["stable_allocation_contrasts"]["H1_primary_MB_vs_MBT"]),
            ("One contrast/experiment", rob["one_contrast_per_experiment"]["H1_primary_MB_vs_MBT"]),
            ("Held-out experiments", rob["experiment_heldout"]["H1_primary_MB_vs_MBT"]),
            ("Gradient-boosted learner", rob["flexible_gbm"]["H1_flex_MB_vs_MBT"])]
    fig, ax = plt.subplots(1, 2, figsize=(W, 3.0), gridspec_kw={"width_ratios": [1, 1]})
    s = 1e4
    y = np.arange(len(rows))[::-1]
    for yi, (lab, v) in zip(y, rows):
        if v.get("degenerate_identical_predictions"):
            ax[0].plot(0, yi, "o", mfc="white", mec=BLUE, ms=6, mew=1.4)
        else:
            ax[0].errorbar(v["estimate"] * s, yi, xerr=[[(v["estimate"] - v["ci_low"]) * s], [(v["ci_high"] - v["estimate"]) * s]],
                           fmt="o", color=BLUE, ms=6, capsize=2, lw=1.4)
    ax[0].axvline(r["h1_decision"]["null_q975"] * s, color=ORANGE, lw=1, ls="--")
    ax[0].text(r["h1_decision"]["null_q975"] * s + 0.1, y[-1] - 0.9, "null 97.5th pct", color=INK2, fontsize=10)
    zero_line(ax[0])
    ax[0].set_yticks(y, [x[0] for x in rows])
    ax[0].set_ylim(-1.3, len(rows) - 0.4)
    ax[0].set_xlabel("ΔR(MB→MBT) × 10⁴, 95% CI")
    ax[0].set_title("(a) Primary and robustness")
    fp = pd.read_csv(RES / "exploratory_fixed_penalty.csv")
    fp = fp[np.isinf(fp.lam2)]
    for comp, col, lab in [("M0_vs_MB", INK2, "M0→MB"), ("MB_vs_MBT", BLUE, "MB→MBT"), ("MS_vs_MST", ORANGE, "MS→MST")]:
        d = fp[fp.comparison == comp].sort_values("lam1")
        ax[1].plot(d.lam1, d.estimate * s, "-o", color=col, ms=5, lw=2, label=lab)
        if comp == "MB_vs_MBT":  # one band only, for legibility
            ax[1].fill_between(d.lam1, d.ci_low * s, d.ci_high * s, color=col, alpha=0.15, lw=0, label="95% CI (MB→MBT)")
    zero_line(ax[1], vertical=False)
    ax[1].set_xscale("log")
    ax[1].set_xlabel("Shared-slope penalty λ₁ (forced)")
    ax[1].set_ylabel("ΔR × 10⁴ (<0 = worse)")
    ax[1].set_title("(b) Exploratory: forced feature use")
    ax[1].legend(loc="lower right", fontsize=10, handlelength=1.5)
    ax[1].set_ylim(-4.6, 0.3)
    fig.tight_layout(w_pad=1.5)
    save(fig, "fig_main")


def fig_prognosis():
    d = pd.read_csv(RES / "exploratory_prognosis_vs_benefit.csv")
    cols = {"s": BLUE, "b_correct_all": ORANGE, "t_slope_correct_norm": AQUA, "t_recent_shift_correct": YELLOW}
    labels = {"s": "Prognostic score", "b_correct_all": "Lifetime correctness", "t_slope_correct_norm": "Correctness slope",
              "t_recent_shift_correct": "Recent shift"}
    fig, ax = plt.subplots(1, 2, figsize=(W, 2.9), sharex=True)
    for k, (axis, c) in enumerate(cols.items()):
        g = d[d.axis == axis]
        off = (k - 1.5) * 0.09
        ax[0].errorbar(g.quintile + off, 100 * g.completion, yerr=196 * g.completion_se, fmt="-o", color=c, ms=5, lw=2,
                       capsize=0, label=labels[axis])
        ax[1].errorbar(g.quintile + off, 100 * g.benefit, yerr=196 * g.benefit_se, fmt="-o", color=c, ms=5, lw=2, capsize=0)
    zero_line(ax[1], vertical=False)
    ax[0].set_ylabel("Completion rate (%)")
    ax[0].set_title("(a) Prognosis: who completes")
    ax[1].set_ylabel("Benefit of better arm (pp)")
    ax[1].set_title("(b) Responsiveness: who benefits")
    for a in ax:
        a.set_xticks(range(1, 6), ["Q1", "Q2", "Q3", "Q4", "Q5"])
        a.set_xlabel("Quintile of the feature")
    fig.tight_layout(w_pad=1.5, rect=(0, 0.1, 1, 1))
    fig.legend(*ax[0].get_legend_handles_labels(), loc="lower center", ncol=4, fontsize=10, bbox_to_anchor=(0.5, 0.0))
    save(fig, "fig_prognosis")


def fig_itt():
    itt = pd.read_csv(RES / "contrast_itt.csv")
    b = pd.read_csv(RES / "exploratory_blp_by_feature.csv")
    fig, ax = plt.subplots(1, 2, figsize=(W, 2.9), gridspec_kw={"width_ratios": [1, 1.25]})
    o = itt.sort_values("ate_hist").reset_index(drop=True)
    sig = (o.ate_hist - 1.96 * o.se_hist > 0) | (o.ate_hist + 1.96 * o.se_hist < 0)
    ax[0].vlines(o.index, 100 * (o.ate_hist - 1.96 * o.se_hist), 100 * (o.ate_hist + 1.96 * o.se_hist),
                 color=np.where(sig, BLUE, GRID), lw=0.8)
    ax[0].plot(o.index, 100 * o.ate_hist, ".", color=INK, ms=2)
    zero_line(ax[0], vertical=False)
    ax[0].set_xlabel("259 contrasts, sorted")
    ax[0].set_ylabel("ATE on completion (pp)")
    ax[0].set_title("(a) Effects differ between contrasts")
    short = {"b_log_n_prior": "n prior", "b_log_n_prior_stats": "n prior (stats)", "b_log_tenure_days": "tenure",
             "b_log_recency_days": "recency", "b_completion_rate_all": "completion (life)", "b_correct_all": "correct (life)",
             "b_correct_norm_win": "correct (norm.)", "b_ln_time_on_task_win": "time on task", "b_ln_first_resp_win": "first resp.",
             "b_attempts_win": "attempts", "b_hints_win": "hints", "b_answer_given_win": "answer req.",
             "b_problems_per_assign_win": "problems/assign.", "b_skill_builder_share_win": "skill-builder share",
             "b_active_days_60": "active days", "b_has_due_date": "due date", "t_slope_correct_norm": "T: correct slope",
             "t_resid_sd_correct_norm": "T: variability", "t_recent_shift_correct": "T: recent shift",
             "t_completion_shift": "T: completion shift", "t_slope_hints_norm": "T: hint slope",
             "t_activity_trend": "T: activity trend", "t_gap_cv": "T: gap regularity"}
    b = b[b.feature.str.startswith(("b_correct", "b_completion", "b_hints", "b_attempts", "b_active", "b_ln_time", "t_"))]
    yy = np.arange(len(b))[::-1]
    istraj = b.feature.str.startswith("t_").to_numpy()
    ax[1].errorbar(b.coef_pp_per_sd, yy, xerr=1.96 * b.se_pp, fmt="none", ecolor=GRID, lw=1.4)
    ax[1].scatter(b.coef_pp_per_sd, yy, c=np.where(istraj, BLUE, ORANGE), s=22, zorder=3)
    zero_line(ax[1])
    ax[1].set_yticks(yy, [short[f] for f in b.feature], fontsize=10)
    ax[1].set_xlabel("Effect change per SD (pp), 95% CI")
    ax[1].scatter([], [], c=ORANGE, s=22, label="level")
    ax[1].scatter([], [], c=BLUE, s=22, label="trajectory")
    ax[1].legend(loc="lower right", fontsize=10, handletextpad=0.2)
    ax[1].set_title("(b) Exploratory per-feature moderation")
    fig.tight_layout(w_pad=1.2)
    save(fig, "fig_itt_features")


def fig_decomposition():
    dec = json.loads((RES / "decomposition_results.json").read_text())
    rob = json.loads((RES / "randomization_robustness.json").read_text())
    pr = dec["primary"]
    fig, ax = plt.subplots(1, 2, figsize=(W, 2.9), gridspec_kw={"width_ratios": [1.45, 1]})
    rows = [("Intervention (between contrasts)", 100 * rob["between_all"]["diff_in_means"]["tau_sd_across_contrasts"], None,
             [100 * rob["between_all"]["dr_adjusted"]["tau_sd_across_contrasts"], 100 * rob["between_clean"]["dr_adjusted"]["tau_sd_across_contrasts"]]),
            ("Student history (CATE models)", 0.0, None, []),
            ("Context total", pr["total_context"]["sd_pp"], pr["total_context"]["sd_upper_pp"], []),
            ("Context total, clean subset", rob["decomposition_clean"]["total_context"]["sd_pp"],
             rob["decomposition_clean"]["total_context"]["sd_upper_pp"], []),
            ("   classroom", pr["class"]["sd_pp"], pr["class"]["sd_upper_pp"], []),
            ("   teacher", pr["teacher"]["sd_pp"], pr["teacher"]["sd_upper_pp"], []),
            ("   district", pr["district"]["sd_pp"], pr["district"]["sd_upper_pp"], [])]
    y = np.arange(len(rows))[::-1]
    for yi, (lab, pt, up, extra) in zip(y, rows):
        col = BLUE if lab.startswith("Intervention") else (INK2 if lab.startswith("Student") else ORANGE)
        if up is not None:
            ax[0].plot([0, up], [yi, yi], color=col, lw=2.2, alpha=0.35, solid_capstyle="butt")
            ax[0].plot(up, yi, "|", color=col, ms=10, mew=2)
        ax[0].plot(pt, yi, "o", color=col, ms=7, zorder=3)
        for e in extra:
            ax[0].plot(e, yi, "o", mfc="white", mec=col, ms=6, mew=1.4, zorder=3)
    ax[0].text(0.15, y[1] + 0.3, "ΔR = 0 (not detected)", color=INK2, fontsize=10)
    ax[0].set_yticks(y, [r[0] for r in rows])
    ax[0].set_xlim(-0.2, 7.5)
    ax[0].set_xlabel("SD of effects on completion (pp)")
    ax[0].set_title("(a) Where effect variation lives")
    dp = pd.read_csv(AUDIT / "design_projection.csv")
    ms = dp.m.to_numpy(float)
    beta = -np.polyfit(np.log(ms), np.log(dp.se_total), 1)[0]
    grid = np.linspace(1, 6, 60)
    se_g = dp.se_total.iloc[0] * grid ** (-beta)
    from scipy import stats as st
    tv = dp.true_V_alt.mean()
    ax[1].plot(grid, st.norm.cdf(tv / se_g - 1.96), color=ORANGE, lw=2, label="projected")
    ax[1].plot(ms, dp.power_sim_alt, "o", color=ORANGE, ms=7, label="simulated")
    ax[1].axhline(0.8, color=INK2, lw=0.8, ls="--")
    ax[1].set_ylim(0, 1.02)
    ax[1].set_xlabel("Students per class (× current)")
    ax[1].set_ylabel("Power")
    ax[1].set_title("(b) Detecting 3.4 pp context SD")
    ax[1].legend(loc="lower right", fontsize=10)
    fig.tight_layout(w_pad=1.2)
    save(fig, "fig_decomposition")


def fig_leakage():
    t = pd.read_csv(RES / "data_gaps_leakage.csv")
    feats = ["b_completion_rate_all", "b_correct_norm_win", "t_completion_shift", "t_recent_shift_correct"]
    names = ["Lifetime\ncompl.", "Window\ncorrect", "Compl.\nshift", "Correct\nshift"]
    srcs = [("clean", "Clean (≥24 h cut)", BLUE), ("leak_self", "Includes target row", ORANGE), ("leak_all", "Includes future rows", YELLOW)]
    fig, ax = plt.subplots(1, 2, figsize=(W, 2.6))
    wdt = 0.26
    for k, (src, lab, col) in enumerate(srcs):
        g = t[t.source == src].set_index("feature").loc[feats]
        x = np.arange(len(feats)) + (k - 1) * wdt
        ax[0].bar(x, g.auc_completion, width=wdt - 0.03, color=col, label=lab)
        ax[1].bar(x, g.Q0 / g.df, width=wdt - 0.03, color=col)
    ax[0].set_ylim(0.45, 0.78)
    ax[0].set_ylabel("AUC for completion")
    ax[0].set_title("(a) Leakage inflates prediction")
    ax[1].axhline(1, color=INK2, lw=0.8, ls="--")
    ax[1].set_ylabel("Arm imbalance (Q₀ / df)")
    ax[1].set_title("(b) …and adds arm imbalance")
    for a in ax:
        a.set_xticks(range(len(feats)), names, fontsize=10)
    fig.tight_layout(w_pad=1.2, rect=(0, 0.1, 1, 1))
    fig.legend(*ax[0].get_legend_handles_labels(), loc="lower center", ncol=3, fontsize=10, bbox_to_anchor=(0.5, 0.0))
    save(fig, "fig_leakage")


if __name__ == "__main__":
    fig_simulation()
    fig_main()
    fig_prognosis()
    fig_itt()
    fig_leakage()
    if (AUDIT / "design_projection.csv").exists():
        fig_decomposition()
    print("report figures written to", OUT)
