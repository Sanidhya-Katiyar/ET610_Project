# Protocol: where does treatment-effect variation live? (frozen before computation)

**Context.** The confirmatory study (`protocol/ANALYSIS_PROTOCOL.md`) found:
- no detectable within-contrast heterogeneity explained by student history (ΔR = 0);
- clear between-contrast variation in average effects: DerSimonian–Laird SD 3.3 pp in the primary sample, 3.6 pp among all randomized units.

This protocol adds the missing **context** component. It was written after the literature check (`reports/DECOMPOSITION_LIT_CHECK.md`) and the outcome-blind simulation (`reports/DECOMPOSITION_SIMULATION.md`). No class-, teacher- or district-level outcome statistic by arm has been computed before this freeze.

**Question.** How much do the effects of instructional variants on completion vary across districts, teachers and classrooms within the same randomized contrast?

## 1. Data

- **Population (primary):** all randomized units in the 259 primary contrasts: `data_access.load_frame("b24_k20", all_randomized=True)`, 219,760 unit-contrast rows. History-based features may be missing; they are median-imputed within training folds, and missing prognostic scores are set to the median.
- **Outcome:** `assignment_completed`.
- **Context:**
  - class from the target row; teacher and district from the E-TRIALS covariates file, linked by (experiment, student);
  - strict nesting enforced: each class takes its modal teacher, each teacher its modal district;
  - units with no district ID put each teacher in their own pseudo-district (about 21%), which makes the district component conservative.
  - Built by `src/12_decomposition_simulation.py::build_frame`.

## 2. Estimator

- **DR pseudo-outcomes:** `pipeline.crossfit` with models = (M0,), 5 folds grouped by connected components of the student–district graph (`ctx_group`), nuisances as in the main protocol (arm-specific ridge LPM, λ = 10, contrast × arm fixed effects), and p_c from training folds.
- **Pair-product decomposition** (`src/decomposition.py`):
  - V_class = M_same-class − M_same-teacher-different-class;
  - V_teacher = M_same-teacher-different-class − M_same-district-different-teacher;
  - V_district = M_same-district-different-teacher − M_different-district.
  - Each is computed within contrast and pooled with weights n_c.
- **Standard errors:** leave-one-district-out jackknife.
- **Repetitions:** 5 cross-fitting seeds (20261029 + 101·r). Point estimates and SEs are averaged across seeds; averaging SEs is conservative.

## 3. Primary estimand and test

- **Estimand:** V_total = V_class + V_teacher + V_district, the variance of the context component of contrast-level effects. It is reported as an SD in pp, √max(V, 0).
- **Primary test:** context variation is **detected** if the jackknife 95% CI for V_total lies entirely above 0 (one-sided α = 0.025). Simulated false-positive rate: 0.02.
- **If not detected,** report the 95% upper bound on the context SD, √max(0, V̂ + 1.96·SE). The simulation suggests about 4.7 pp at V̂ = 0.
- **Pre-specified comparison (descriptive):** the context SD (estimate and upper bound) against the between-contrast SD of average effects among all randomized units (3.6 pp).

**Interpretation rules (fixed now):**
- Detected, with a point SD ≥ 3.6 pp: "context moderation is at least comparable to differences between interventions."
- Detected, with a point SD < 3.6 pp: "context moderation is present but smaller than intervention differences."
- Not detected: "context moderation is below [upper bound] pp; we cannot rule out context variation of the size of intervention differences" when the upper bound is ≥ 3.6 pp.

## 4. Secondary and exploratory (no confirmatory claims)

1. V_class, V_teacher and V_district separately, with jackknife CIs. Labelled exploratory, because the simulation shows low power per level.
2. Sensitivity: the same primary test in the ≥10-history primary sample (147,704 rows).
3. Sensitivity: only units with a real district ID; units in pseudo-districts are dropped. A dry run on fake outcomes showed that only 1 contrast has real district IDs for every unit, so the contrast-level restriction first planned was replaced before the freeze.

## 5. Assumptions and threats (stated in advance)

- No interference between classmates. Spillovers would load onto the class component.
- Context IDs are pre-treatment.
- Pseudo-districts bias V_district downwards.
- Students in several contrasts contribute to several contrast-level estimates. The jackknife over districts accounts for dependence within districts.

## 6. Run once

`src/13_decomposition_analysis.py` checks this file's SHA-256 against `protocol/DECOMPOSITION_PROTOCOL_HASH.txt` and is run once. Any later change is logged as a deviation.
