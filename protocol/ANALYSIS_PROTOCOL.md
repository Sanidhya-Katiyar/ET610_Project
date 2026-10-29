# Analysis protocol (frozen before any treatment-by-outcome analysis)

**Title:** *Prognostic but Not Predictive? Do Pre-Treatment Learning Trajectories Add Information About Who Benefits Across Randomized Online-Math Experiments?*

**Freeze status:** frozen at 2026-10-29T13:20:38+0530 (local time).
- Local, timestamped and SHA-256-hashed freeze (`protocol/PROTOCOL_HASH.txt`). **This is not a public preregistration.** No OSF registration was submitted: that needs the owner's account and authorization.
- The analysis code (`src/05_primary_analysis.py`) refuses to run unless this file's hash matches.
- Written after the outcome-blind audit (`reports/OUTCOME_BLIND_AUDIT.md`) and the outcome-blind simulation (`reports/audit_tables/simulation_summary.csv`). Before the freeze, no experimental outcome was tabulated, modelled or plotted by arm.

## 1. Data and population

- **Data:** public OSF project k8ph9 (Sales et al., 2023, JEDM 15(2):53–85), ASSISTments Data Terms of Use.
- **Randomized unit:** a student's first start of an experimental skill builder whose E-TRIALS `assigned_condition` is a real arm. Excluded: "Not Assigned" rows, the 36 assignment logs listed under two conditions, and the 10 rows whose released flags contradict their label.
- **Contrasts:** every unordered pair of arms within an experiment. z = 1 for the arm whose label sorts later, for example Treatment vs. Control or Treatment 2 vs. Treatment 1.
- **Primary target population:** randomized units with ≥ 10 prior assignments with problem statistics, started ≥ 24 h before the target start. They must belong to contrasts with ≥ 56 such units in each arm. Outcome-blind counts: **259 contrasts, 63 experiments, 147,704 unit-contrast rows, 60,439 units, 26,582 students.**
- **Population check:** all randomized units of the same contrasts, with no history rule.
- **Scope of pooled results:** they average over this distribution of heterogeneous, mostly undocumented instructional variants. They are not effects of one named intervention.

## 2. Assignment probability

Within contrast c, Z is Bernoulli with a constant probability p_c (independent, one-at-a-time E-TRIALS randomization). p_c is estimated by the arm share among the contrast's training-fold rows. Some contrasts deviate from ½ but are stable over time; see the audit. **No contrast is excluded because of a p-value on its realized allocation.**

## 3. Outcome

- **Primary:** `assignment_completed` (binary; reached mastery in the target skill builder). It is the ITT outcome, observed for every randomized unit. Completion is a composite of persistence, engagement and learning, not a pure measure of knowledge.
- **Exploratory only:** `problems_completed`. It counts problems attempted for non-completers and problems-to-mastery for completers.

## 4. Features (`reports/FEATURE_DICTIONARY.md`)

- **X_B (16 level/practice features)** and **X_T (7 trajectory features)**, rebuilt from raw timestamped history.
- **Primary window:** the last 20 eligible assignments with statistics.
- Winsorized at the remnant 1st/99th percentiles. Within each training fold: median imputation, then standardization.
- **Prognostic score s:** logit from a gradient-boosted model trained on remnant students only (class-grouped CV AUC 0.704).

## 5. Models (identical tuning for every model)

CATE model: τ_c(x) = α_c + x′(β + b_c).
- α_c is the training-fold contrast mean of the DR pseudo-outcome.
- β are shared slopes with ridge penalty λ1 ∈ {∞, 3e4, 3e3, 3e2, 3e1}.
- b_c are contrast-specific deviations with ridge penalty λ2 ∈ {∞, 3e3, 3e2, 3e1}.
- (λ1, λ2) are chosen by inner 3-fold group CV on the pseudo-outcome loss, within training folds only.

| Model | Moderators x |
|---|---|
| M0 | none (contrast ATE only) |
| MB | X_B |
| **MBT** | **X_B + X_T** |
| MS | s |
| MST | s + X_T |

- **Nuisances:** arm-specific linear probability models with contrast × arm fixed effects and ridge slopes (λ = 10) on X_B, X_T and s.
- **Training pseudo-outcomes:** nuisances inner-cross-fitted (2 group folds).
- **Held-out pseudo-outcomes:** nuisances fitted on all training-fold rows.

## 6. Primary estimand and test

- **Folds:** 5-fold cross-fitting with folds formed from connected components of the student–class graph (2,794 groups, largest 1.8% of students). All rows of a student stay in one fold.
- **DR score:** φ_i = μ̂1 − μ̂0 + Z(Y − μ̂1)/p_c − (1 − Z)(Y − μ̂0)/(1 − p_c). The nuisances and p_c come from training folds only.
- **Identification:** because Z ⟂ (Y(0), Y(1), X) within a contrast with constant p_c, E[φ_i | X_i, training data] = τ_c(X_i) for any nuisances. So E[(φ − τ̂(X))²] = E[(τ − τ̂(X))²] + a term that does not depend on τ̂. Paired held-out loss differences against the same φ are therefore unbiased for differences in τ-risk; the full derivation is in `src/estimation.py`. Individual effects are never treated as observed.
- **Estimand:** ΔR = (1/N) Σ_i [(φ_i − τ̂_MB(x_i))² − (φ_i − τ̂_MBT(x_i))²] over held-out unit-contrast rows (student-contrast weights). Positive values mean the trajectory features improve CATE estimation.
- **Inference:** the cluster-robust SE of the mean uses the same groups. The point estimate and SE are averaged over **5 cross-fitting repetitions** (seeds 20261029 + 101·r). Averaging SEs is conservative.
- **H1 (primary):** supported only if **both** (i) the two-sided 95% CI for ΔR lies entirely above 0, and (ii) ΔR̂ exceeds the 97.5th percentile of the simulated null distribution of ΔR̂ (scenarios a and b pooled, 200 replicates; value in §10). Otherwise H1 is not supported. Criterion (ii) was added before the freeze because the simulation showed the CI criterion alone rejects in 8/100 pure-null replicates (§10).
- **Practical equivalence (SESOI):**
  - The SESOI is trajectory-driven heterogeneity of **≥ 3 percentage points of completion effect per SD of the trajectory index**, scaled by arm position (simulation scenario d).
  - Substantive interpretation: under that heterogeneity, an oracle trajectory-based arm choice would change completion by about 1 pp relative to the best fixed arm.
  - The simulated distribution of ΔR̂ under scenario d (`simulation_reps.csv`) is the reference. If the observed ΔR̂ falls below that distribution's 5th percentile (model-based p_SESOI < 0.05), SESOI-sized trajectory heterogeneity is rejected. This is model-based: it is conditional on the simulation's data-generating process.
  - The simulation uses 1 repetition per replicate, while the observed estimate averages 5. The observed estimate is less variable, so this comparison is conservative.

## 7. Secondary confirmatory tests (Holm-adjusted across S1–S3; one-sided, α = 0.05)

- **S1 (trajectory beyond prognosis):** ΔR(MS → MST) > 0.
- **S2 (decision value):** the held-out AIPW value of the policy 1[τ̂_MBT > 0] minus that of 1[τ̂_MB > 0], within contrasts, is > 0.
- **S3 (any heterogeneity detectable from level features):** ΔR(M0 → MB) > 0.

**Descriptive comparisons (not tested):**
- ΔR for M0 → MBT, M0 → MS and MS → MBT;
- policy gains of MB and MBT over the best fixed arm per contrast (fixed arm = sign of the training-fold contrast ATE);
- a BLP-type increment coefficient (regress φ on τ̂_MB and τ̂_MBT − τ̂_MB within contrasts);
- within-contrast SDs of the predicted CATEs.

## 8. Pre-specified robustness analyses (each reports ΔR(MB → MBT) with a 95% CI)

1. **Equal-contrast weights:** each contrast weighted 1/n_c.
2. **K = 10 window,** ≥ 5 prior assignments (283 contrasts).
3. **Strict 60 s cutoff** instead of 24 h (278 contrasts).
4. **p_c = ½** fixed, and p_c = full-contrast arm share.
5. **One contrast per experiment** (the largest, an outcome-blind choice).
6. **Held-out experiments:** folds grouped by experiment, p_c = full-contrast arm share. This measures transfer to unseen contrasts in the corpus only.
7. **Stable-allocation subset:** drops the 4 contrasts whose allocation varied over calendar time (χ² p < 0.001).
8. **Flexible learner:** a gradient-boosted DR-learner (contrast intercepts plus boosting with an experiment categorical) replaces the ridge models for MB and MBT.
9. **Permutation placebo:** Z permuted within contrasts, 5 draws. ΔR should not be positive.
10. **Descriptive randomization checks:** standardized mean differences of all features by arm. Arm shares are reported, not used as filters.
11. **ITT:** contrast-level difference-in-means ATEs in the primary sample and in all randomized units, and DerSimonian–Laird between-contrast heterogeneity. Contrast orientation is arbitrary, so the mean ATE is not interpreted.

**Exploratory (labelled as such):**
- per-feature best linear projections of τ, in Control-vs-Treatment contrasts only;
- `problems_completed` ITT;
- anything added after unblinding.

## 9. Fallbacks and deviations

- Any change after the freeze will be logged in `reports/RESULTS.md` under "Deviations" with a reason.
- If cross-fitting or estimation fails for a robustness variant, that is reported, not dropped.
- No result is selected by significance.

## 10. Outcome-blind simulation results relied upon

All values come from `reports/audit_tables/simulation_summary.csv` and `simulation_reps.csv` (`src/04_simulation.py`).
Setup: 100 replicates per scenario, 1 cross-fitting repetition each. Synthetic completion outcomes are drawn on the **real** primary sample (covariates, contrasts, arm assignments, groups); no experimental outcome is used. ΔR is in squared-probability units.

| Scenario | True within-contrast SD of τ | Mean ΔR̂ (MB → MBT) | Bias | Emp. SD | Mean SE | Coverage of realized truth | P(CI lower > 0) ± MC SE |
|---|---|---|---|---|---|---|---|
| a: null (no heterogeneity) | 0.0004 | 0.000004 | 0.000001 | 0.000058 | 0.000019 | 0.99 | 0.08 ± 0.027 |
| b: level-only heterogeneity | 0.046 | −0.000051 | 0.000000 | 0.000099 | 0.000069 | 0.88 | 0.02 ± 0.014 |
| c: trajectory, 1.5 pp/SD | 0.026 | 0.000516 | −0.000012 | 0.000163 | 0.000152 | 0.95 | 0.92 ± 0.027 |
| d: trajectory, 3 pp/SD (SESOI) | 0.049 | 0.002187 | −0.000017 | 0.000290 | 0.000274 | 0.93 | 1.00 |
| e: trajectory, mixed signs across contrasts | 0.023 | −0.000039 | −0.000014 | 0.000099 | 0.000039 | 0.99 | 0.02 ± 0.014 |
| f: trajectory, 6 pp/SD | 0.090 | 0.007284 | 0.000062 | 0.000502 | 0.000566 | 0.99 | 1.00 |

Secondary tests (P(CI lower > 0)):
- S1 (MS → MST): a 0.03, b 0.00, c 0.94, d 1.00, e 0.17, f 1.00.
- S2 (policy MBT − MB): a 0.04, b 0.00, c 0.88, d 1.00, e 0.03, f 1.00.
- S3 (M0 → MB): 1.00 in b (level heterogeneity) and 0.00 in every other scenario.

**What this implies for the frozen decisions:**
- The ΔR estimator is essentially unbiased in every scenario (|bias| < 1% of the SESOI-scenario ΔR).
- Power at the SESOI (scenario d) is 1.00, and 0.92 at half the SESOI (scenario c).
- **Type-I error.** In pure-null scenario a, the CI criterion rejected in 8/100 replicates (MC 95% interval ≈ 3–13%), above the nominal 2.5%. In all 8, the realized true ΔR of the fitted models was positive, so these are rejections of a population null, not of the realized-model estimand. Scenario b rejected 2/100. Hence the added criterion (ii) of H1: **null 97.5th percentile of ΔR̂ = 0.000142** (scenarios a + b, 200 replicates). The observed analysis averages 5 repetitions, which lowers its variance, so comparing it with single-repetition null draws is conservative.
- **SESOI reference.** Scenario d: mean ΔR̂ 0.002187, **5th percentile 0.001787**. An observed ΔR̂ below 0.001787 rejects SESOI-sized trajectory heterogeneity (model-based, §6).
- **Limitation.** Scenario e (sign-mixed trajectory heterogeneity across contrasts) is essentially undetectable by H1 (true ΔR ≈ 0 with the frozen shrinkage grid). A null H1 therefore does not rule out contrast-specific trajectory moderation with signs that cancel; S1 detected it in 17% of replicates.
- Coverage is 0.88–0.99. The lowest value (b, 0.88) occurs where trajectories carry no signal.

