# Where does variation in the effects of instructional changes live?

Reproducible analysis code for an ET610 group project: a re-analysis of 63 randomized E-TRIALS experiments (259 pairwise contrasts) in the ASSISTments online math platform.

It asks where variation in the effects of instructional changes on assignment completion comes from:
- **the intervention itself**;
- **the student's history** (prior performance levels and recent learning trajectories);
- **the context** (district, teacher, classroom).

It also asks what data would be needed to see these sources.

**Main results:**

| Source of effect variation | Finding |
|---|---|
| Intervention (between contrasts) | Detected: SD 3.1–3.6 percentage points (pp) |
| Student history: 16 level features, 7 trajectory features, a remnant-trained prognostic score | Not detected: cross-validated CATE models collapse to constant effects (ΔR = 0), although histories predict completion well |
| Context (district + teacher + classroom) | Not detected: point SD 2.1 pp, 95% upper bound 5.0 pp |

An outcome-blind audit of the public release found three problems, all corrected here:
- the treatment flags merge arms;
- the "prior" history leaks the target assignment and later ones;
- the precomputed covariates do not reproduce.

Exploratory analyses quantify what each data gap costs.

## Data (not included)

The data are the public replication files of:

> Sales, A. C., Prihar, E. B., Gagnon-Bartsch, J. A., & Heffernan, N. T. (2023). Using auxiliary data to boost precision in the analysis of A/B tests on an online educational platform: New data and new results. *Journal of Educational Data Mining, 15*(2), 53–85. https://doi.org/10.5281/zenodo.8016854. Data: https://osf.io/k8ph9/

The data are governed by the **ASSISTments Data Terms of Use** (in the OSF project's `license.txt`). They must not be redistributed, so this repository contains **no data and no derived files**. Downloading the files means accepting those terms: cite the dataset, no commercial use, no re-identification, and open-access reporting of all outcomes. Each user must download the data themselves with `src/00_download.py`, which also verifies every file against the OSF SHA-256 checksums.

## Setup

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate   |   macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
```

Tested with Python 3.11.9 on Windows 11, with 16 threads and 16 GB RAM. You need about 6 GB of free disk space. DuckDB is capped at 6 GB of memory (`src/common.py`).

## Reproduce everything

Run from the repository root, in this order. Every script reads the outputs of the earlier ones.

```bash
# 0. Data
python src/00_download.py                    # ~2.3 GB from OSF -> data/raw/osf_k8ph9/, checksummed
python -c "import zipfile; zipfile.ZipFile('data/raw/osf_k8ph9/data/covariates.zip').extractall('data/raw/osf_k8ph9')"

# 1. Outcome-blind preparation (no outcome is compared between arms)
python src/01_outcome_blind_audit.py         # schema, arms, leakage and linkage audit
python src/02_build_sample.py                # randomized units, pairwise contrasts, features (3 variants)
python src/03_remnant_score.py               # prognostic score trained on remnant students only
OMP_NUM_THREADS=1 python src/04_simulation.py --reps 100 --jobs 14     # synthetic outcomes: power, SESOI

# 2. Confirmatory test of H1-H4 (refuses to run unless protocol/ANALYSIS_PROTOCOL.md matches its hash)
python src/05_primary_analysis.py
python src/06_figures.py

# 3. Exploratory follow-ups to H1-H4
python src/07_exploratory_fixed_penalty.py   # forced feature use
python src/08_exploratory_prognosis_vs_benefit.py

# 4. Follow-up feasibility and context decomposition (H5)
python src/10_followup_feasibility.py        # outcome-blind audit of post-target rows
python src/11_followup_selection_check.py    # frozen selection check (protocol/FOLLOWUP_SELECTION_CHECK.md)
OMP_NUM_THREADS=1 python src/12_decomposition_simulation.py --jobs 6                       # primary sample
OMP_NUM_THREADS=1 python src/12_decomposition_simulation.py --population all --jobs 6      # all randomized units
python src/13_decomposition_analysis.py      # frozen H5 analysis (protocol/DECOMPOSITION_PROTOCOL.md)

# 5. Exploratory data-gap and robustness analyses
python src/14_data_gap_analyses.py           # merged arms, leakage, released covariates, descriptor power
OMP_NUM_THREADS=1 python src/15_design_projection_sim.py --jobs 3   # power vs. students per class
python src/16_randomization_robustness.py    # arm clustering, calendar balance, clean-subset re-estimation

# 6. Report figures (written to report/figures and report/overleaf/figures)
python src/09_report_figures.py

# Tests (synthetic fixtures only; no data needed)
python -m pytest -q tests
```

On Windows PowerShell, set environment variables with `$env:OMP_NUM_THREADS=1` before the command.

**Approximate run times** on a 16-thread laptop:

| Step | Time |
|---|---|
| Download | 3 min |
| Audit and sample build | 7 min |
| `04` simulation | about 70 min with 14 workers |
| `05` analysis | 5 min |
| `12` simulation | about 30–40 min per population |
| `14` data-gap analyses | 7 min |
| `15` design projection | about 20 min |

The simulations are memory-hungry. If the machine runs low on memory, lower `--jobs`; 3–6 workers worked with 16 GB.

**Outputs** go to:
- `reports/audit_tables/`: audit tables and simulation summaries;
- `reports/results_tables/`: all results as JSON and CSV;
- `figures/` and `report/figures/`: figures;
- `logs/`: logs.

All of these folders are git-ignored.

## Pre-specification

There are three frozen protocols in `protocol/`, each stored with its SHA-256 hash and the time it was frozen:

| Protocol | Covers | Checked by |
|---|---|---|
| `ANALYSIS_PROTOCOL.md` | H1–H4 | `src/05` |
| `FOLLOWUP_SELECTION_CHECK.md` | follow-up selection check | `src/11` |
| `DECOMPOSITION_PROTOCOL.md` | H5 | `src/13` |

Each checking script refuses to run if its protocol file has been modified. The freezes are local, not public preregistrations. Analyses in `07`, `08`, `14`, `15` and `16` were added after unblinding and are exploratory.

`.gitattributes` keeps the protocol files byte-exact (no line-ending conversion), so the hash checks also pass on other operating systems.

## Code map

| File | Purpose |
|---|---|
| `src/common.py` | Paths, seed (`20261029`), DuckDB views over the raw CSVs |
| `src/features.py` | Time-cutoff SQL and window features (level vs. trajectory) |
| `src/data_access.py` | Outcome-free analysis frames per feature variant |
| `src/estimation.py` | Folds, DR pseudo-outcomes, pooled ridge CATE model, cluster SEs, policy scores |
| `src/pipeline.py` | One cross-fitted run; τ-risk differences and policy gains |
| `src/decomposition.py` | Pair-product decomposition of effect variation (class / teacher / district) with district jackknife |
| `src/00`–`src/16` | Pipeline steps, as listed above |
| `tests/` | 14 unit tests: cutoffs and leakage, contrast orientation, folds, DR unbiasedness, ridge recovery, outcome join, recovery of injected variance components |

## Determinism

- Seeds are fixed throughout: `common.SEED`, plus documented offsets.
- Results should be identical on the same platform and library versions (`requirements.txt`).
- Small differences are possible across BLAS builds.


