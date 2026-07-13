# OASIS-3 Event-Based Model: Staging Alzheimer's Progression

This project reconstructs the order in which brain regions become abnormal over the course of Alzheimer's disease, using an **Event-Based Model (EBM)** fit to structural MRI and amyloid-PET data from the [OASIS-3](https://www.oasis-brains.org/) cohort. Rather than sorting people into coarse clinical buckets (normal / impaired / demented), the model assigns every subject a continuous disease **stage**, derived entirely from their biomarkers — and that stage is then validated against real longitudinal cognitive decline.

## What it does

1. Merges five raw OASIS-3 exports (FreeSurfer MRI morphometry, amyloid-PET, clinical diagnosis, cognitive testing, demographics) into one subject-level dataset.
2. Defines a strict cognitively-normal reference group and residualizes every MRI biomarker against it, removing age/sex/head-size effects.
3. Fits two independent Event-Based Models — an **MRI-only** panel (5 biomarkers, largest sample) and an **MRI + amyloid** panel (6 biomarkers, smaller but molecularly more specific) — via mixture modeling and MCMC search over biomarker orderings.
4. Validates each model three ways: a permutation test, a bootstrap positional-variance analysis, and a face-validity check against clinical dementia severity (hard-fails the pipeline if staging doesn't correlate positively with CDR).
5. Externally validates the resulting stage against **future** cognitive trajectories using mixed-effects models, plus conversion and sensitivity analyses.
6. Generates the figures and tables below.

## Key results

| | MRI-only panel | MRI + amyloid panel |
|---|---|---|
| Subjects (healthy / not) | 1,093 (880 / 213) | 712 (632 / 80) |
| Discovered event order | ventricles → hippocampus → entorhinal cortex → fusiform → inferior-temporal | ventricles → fusiform → **amyloid** → hippocampus → entorhinal cortex → inferior-temporal |
| Stage vs. clinical severity (Spearman ρ) | 0.41 | 0.34 |
| Permutation test p-value | < 0.001 | < 0.001 |
| MMSE decline per stage·year | −0.26 (p < .001) | −0.16 (p < .001) |
| Odds ratio, AD diagnosis per stage | 1.95× | 2.22× |

In both panels, average stage rises monotonically from cognitively normal → impaired → demented, and a higher baseline stage independently predicts faster future cognitive decline and higher odds of an AD diagnosis — despite the model never being told anyone's diagnosis while the staging itself was built.

The MRI + amyloid panel's non-CN sample (n=80) falls below this project's own confirmatory threshold and is treated as exploratory (see `docs/cohort_size_warning.md`).

## Repository structure

```
config.py                  Central configuration: paths, column names, thresholds, model parameters
scripts/
  00_extract_merge.py       Merge and QC-filter raw OASIS-3 exports
  01_data_prep.py           Diagnosis derivation, CN reference, residualization, z-scoring
  02_feature_engineering.py Build the two EBM biomarker panels
  03_ebm_staging.py         Core EBM: mixture models, MCMC, permutation test, bootstrap, staging
  04_longitudinal_validation.py  Mixed-effects models, conversion analysis, sensitivity checks
  05_figures.py              Publication figures
utils/                       Shared I/O, logging, and validation helpers
docs/                        Environment snapshot and cohort-size caveats
results/                     Figures, tables, and summary-level outputs (see below)
data/                        Not included — see Data section
```

## Data

This repository does **not** include any OASIS-3 data. Raw imaging and clinical data are distributed under OASIS-3's own data use agreement and cannot be redistributed. To run this pipeline yourself, request access at [oasis-brains.org](https://www.oasis-brains.org/) and place the required exports under `data/raw/` (see `config.py` for expected filenames).

Subject-level derived outputs (per-subject stage assignments, MCMC traces, bootstrap sequences, fitted mixture models) are likewise excluded from `results/`. What is included is aggregate: figures, summary tables, and group-level statistics.

## Running the pipeline

Requires Python 3.11+ and:

```
pandas
numpy
scipy
statsmodels
matplotlib
kde_ebm   # pip install git+https://github.com/ucl-pond/kde_ebm.git
```

With `data/raw/` populated, run the scripts in order:

```
python scripts/00_extract_merge.py
python scripts/01_data_prep.py
python scripts/02_feature_engineering.py
python scripts/03_ebm_staging.py
python scripts/04_longitudinal_validation.py
python scripts/05_figures.py
```

## Method reference

The Event-Based Model follows Fonteijn et al. (2012), *"An event-based model for disease progression and its application in familial Alzheimer's disease and Huntington's disease,"* NeuroImage, using the [`kde_ebm`](https://github.com/ucl-pond/kde_ebm) implementation from the UCL POND group.
