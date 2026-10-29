# Pre-specified selection check for option B (frozen before computation)

**Purpose.** Before any downstream outcome is analysed within follow-up windows, test whether the randomized arm changes *whether a unit has a follow-up window*. A window exists when the student starts a later experimental assignment, because the history file is cut at each student's last experimental target (`reports/FOLLOWUP_FEASIBILITY.md`). This check is computed once.

**Population.** The primary analysis sample: 259 contrasts, `data_access.load_frame("b24_k20")`.

**Selection indicators.** For each unit, "window" is the time from its target start to the student's last experimental target start.
- **S7:** window ≥ 7 days. Restricted to units whose target started ≥ 7 days before the data end, the latest target start in the release.
- **S30:** window ≥ 30 days. Same restriction with 30 days.
- **S_any:** window > 0, i.e., any later experimental target.

**Statistics, per indicator.**
1. Per contrast, the difference in proportions d_c = P(S | z = 1) − P(S | z = 0), with its unpooled SE.
2. **Global test of no selection effect:** Q₀ = Σ_c (d_c / se_c)², compared with χ²(k). This is sign-agnostic, because contrast orientation is arbitrary.
3. **Size of selection effects:** the DerSimonian–Laird between-contrast SD of the true d_c (τ̂), and the mean |d_c| with its expected value under the null (from the SEs).
4. **Implied Lee-bound trimming:** the median over contrasts of |d_c| / max(P(S | z = 0), P(S | z = 1)).

**Decision rule.** The primary indicator for feasibility is S7.
- **Within-window analysis feasible (B):** Q₀ is not rejected at α = 0.05 **and** τ̂ ≤ 0.01, i.e., an SD of true selection effects of at most 1 pp.
- **Feasible only with bounds:** Q₀ is rejected but the median implied trimming is ≤ 5%. Lee bounds would be informative.
- **Not feasible:** otherwise.

S30 and S_any are reported alongside S7 and decided by the same rule.

**Not computed:** no downstream outcome (later completion, correctness or activity counts) is computed in this step.
