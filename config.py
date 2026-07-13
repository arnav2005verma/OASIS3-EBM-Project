from __future__ import annotations
from pathlib import Path
from typing import Final 
PROJECT_VERSION: Final[str] = "3.0.0"
RANDOM_SEED: Final[int] = 42
ROOT: Final[Path] = Path(__file__).resolve().parent
DATA_DIR: Final[Path] = ROOT / "data"
RAW_DIR: Final[Path] = DATA_DIR / "raw"
INTERIM_DIR: Final[Path] = DATA_DIR / "interim"
PROCESSED_DIR: Final[Path] = DATA_DIR / "processed"
RESULTS_DIR: Final[Path] = ROOT / "results"
STAGING_RESULTS_DIR: Final[Path] = RESULTS_DIR / "staging"
LONGITUDINAL_RESULTS_DIR: Final[Path] = RESULTS_DIR / "longitudinal"
SENSITIVITY_RESULTS_DIR: Final[Path] = RESULTS_DIR / "sensitivity"
EXTERNAL_VALIDATION_DIR: Final[Path] = RESULTS_DIR / "external_validation"
FIGURES_DIR: Final[Path] = RESULTS_DIR / "figures"
TABLES_DIR: Final[Path] = RESULTS_DIR / "tables"
LOGS_DIR: Final[Path] = ROOT / "logs"
DOCS_DIR: Final[Path] = ROOT / "docs"
_ALL_PROJECT_DIRS: Final[tuple[Path, ...]] = (
    RAW_DIR, INTERIM_DIR, PROCESSED_DIR,
    STAGING_RESULTS_DIR, LONGITUDINAL_RESULTS_DIR, SENSITIVITY_RESULTS_DIR,
    EXTERNAL_VALIDATION_DIR, FIGURES_DIR, TABLES_DIR,
    LOGS_DIR, DOCS_DIR,
)
def ensure_project_dirs() -> None:
    """Create every directory in '_ALL_PROJECT_DIRS' if it does not exist.
    Idempotent - safe to call at the start of every script's main().
    """
    for directory in _ALL_PROJECT_DIRS:
        directory.mkdir(parents=True, exist_ok=True)
ID_COL: Final[str] = "OASISID"
RAW_FILE_PATHS: Final[dict[str, Path]] = {
    "freesurfer": RAW_DIR / "OASIS3_Freesurfer_output.csv",
    "udsd1": RAW_DIR / "OASIS3_UDSd1_diagnosis.csv",
    "udsb4": RAW_DIR / "OASIS3_UDSb4_cdr.csv",
    "demographics": RAW_DIR / "OASIS3_demographics.csv",
    "centiloid": RAW_DIR / "OASIS3_amyloid_centiloid.csv",
}
FREESURFER_SUBJECT_COL: Final[str] = "Subject"
FREESURFER_SESSION_COL: Final[str] = "MR_session"
FS_QC_PASS_VALUES: Final[frozenset[str]] = frozenset({
    "passed",
    "passed with edits",
})
FS_VERSION_FILTER: Final[str] = "5.3"
UDS_DAYS_COL: Final[str] = "days_to_visit"
UDS_SESSION_LABEL_COL: Final[str] = "OASIS_session_label"
MATCH_TOLERANCE_DAYS_PRIMARY: Final[int] = 180
MATCH_TOLERANCE_DAYS_SENSITIVITY: Final[int] = 365
FS_HIPPOCAMPUS_TOT_COL: Final[str] = "TOTAL_HIPPOCAMPUS_VOLUME"
FS_ENTORHINAL_LH_COL: Final[str] = "lh_entorhinal_thickness"
FS_ENTORHINAL_RH_COL: Final[str] = "rh_entorhinal_thickness"
FS_PARAHIPPOCAMPAL_LH_COL: Final[str] = "lh_parahippocampal_thickness"
FS_PARAHIPPOCAMPAL_RH_COL: Final[str] = "rh_parahippocampal_thickness"
FS_FUSIFORM_LH_COL: Final[str] = "lh_fusiform_thickness"
FS_FUSIFORM_RH_COL: Final[str] = "rh_fusiform_thickness"
FS_INFERIORTEMPORAL_LH_COL: Final[str] = "lh_inferiortemporal_thickness"
FS_INFERIORTEMPORAL_RH_COL: Final[str] = "rh_inferiortemporal_thickness"
FS_PRECUNEUS_LH_COL: Final[str] = "lh_precuneus_thickness"
FS_PRECUNEUS_RH_COL: Final[str] = "rh_precuneus_thickness"
FS_POSTERIORCINGULATE_LH_COL: Final[str] = "lh_posteriorcingulate_thickness"
FS_POSTERIORCINGULATE_RH_COL: Final[str] = "rh_posteriorcingulate_thickness"
VENTRICULAR_VOL_COMPONENTS: Final[list[str]] = [
    "Left-Lateral-Ventricle_volume",
    "Right-Lateral-Ventricle_volume",
    "Left-Inf-Lat-Vent_volume", 
    "Right-Inf-Lat-Vent_volume",
]
FS_ICV_COL: Final[str] = "IntraCranialVol"
WHOLE_BRAIN_VOL_COMPONENTS: Final[list[str]] = [
    "CortexVol",
    "SubCortGrayVol",
    "CorticalWhiteMatterVol",
]
FS_QC_STATUS_COL: Final[str] = "FS QC Status"
FS_VERSION_COL: Final[str] = "version"
DEMO_SEX_COL: Final[str] = "GENDER"
DEMO_SEX_RECODE: Final[dict[int, str]] = {1: "M", 2: "F"}
DEMO_AGE_AT_ENTRY_COL: Final[str] = "AgeatEntry"
DEMO_EDUCATION_COL: Final[str] = "EDUC"
DEMO_APOE_COL: Final[str] = "APOE"
APOE_E4_GENOTYPES: Final[frozenset[int]] = frozenset({24, 34, 44})
NORMCOG_COL: Final[str] = "NORMCOG"
DEMENTED_COL: Final[str] = "DEMENTED"
PROBAD_COL: Final[str] = "PROBAD"
POSSAD_COL: Final[str] = "POSSAD"
MCI_FLAG_COLS: Final[list[str]] = [
    "MCIAMEM",
    "MCIAPLUS",
    "MCIAPLAN",
    "MCIAPATT",
    "MCIAPEX",
    "MCIAPVIS",
    "MCINON1",
    "MCINON2",
    "IMPNOMCI",  
]
NON_AD_EXCLUSION_COLS: Final[list[str]] = [
    "VASC",
    "DLB",
    "FTD",
    "PSP",
    "CORT",
    "PRION",
    "HUNT",
]
IMPNOMCI_INCLUDED_IN_CIND: Final[bool] = True
DX_CN: Final[str] = "CN"
DX_CIND: Final[str] = "CIND"
DX_AD: Final[str] = "AD"
DX_OTHER: Final[str] = "OTHER_Dementia"
DX_UNCERTAIN: Final[str] = "UNCERTAIN"
DX_UNCLASSIFIED: Final[str] = "UNCLASSIFIED"
EBM_INCLUDED_DX_GROUPS: Final[list[str]] = [DX_CN, DX_CIND, DX_AD]
MMSE_COL: Final[str] = "MMSE"
CDR_GLOBAL_COL: Final[str] = "CDRTOT"
CDR_SB_COL: Final[str] = "CDRSUM"
MERGED_RAW_FILE: Final[Path] = INTERIM_DIR / "merged_raw.csv"
LONGITUDINAL_RAW_FILE: Final[Path] = INTERIM_DIR / "longitudinal_raw.csv"
COHORT_BASELINE_FILE: Final[Path] = INTERIM_DIR / "cohort_baseline.csv"
EXCLUSION_LOG_FILE: Final[Path] = LOGS_DIR / "exclusion_log.csv"
COHORT_SIZE_WARNING_FILE: Final[Path] = DOCS_DIR / "cohort_size_warning.md"
COHORT_FEATURES_FILE: Final[Path] = PROCESSED_DIR / "cohort_features.csv"
RESIDUALIZATION_MODELS_DIR: Final[Path] = PROCESSED_DIR / "residualization_models"
MIN_FOLLOWUP_VISITS: Final[int] = 1
MIN_FOLLOWUP_DAYS: Final[int] = 180
MIN_N_FOR_EBM: Final[int] = 150
MIN_N_FOR_LONGITUDINAL: Final[int] = 100
ICV_OUTLIER_SD_THRESHOLD: Final[float] = 2.0
CN_DEFINITION_STRICT_PARAMS: Final[dict[str, float]] = {
    "cdrtot_max": 0.0, 
    "mmse_min": 29.0,
}
CN_DEFINITION_LOOSE_PARAMS: Final[dict[str, float]] = {
    "cdrtot_max": 0.0, 
    "mmse_min": 24.0,
}
MMSE_CN_THRESHOLD: Final[float] = CN_DEFINITION_STRICT_PARAMS["mmse_min"]
CDR_CN_THRESHOLD: Final[float] = CN_DEFINITION_STRICT_PARAMS["cdrtot_max"]
MIN_CN_REFERENCE_N: Final[int] = 30 
EVENT_BIOMARKERS: Final[list[str]] = [
    "entrohinal_thickness",
    "hippocampus_vol",
    "parahippocampal_vol", 
    "fusiform_vol",
    "inferiortemporal_vol",
    "precuneus_thickness", 
    "posteriorcingulate_thickness",
    "ventricular_vol",
]
THICKNESS_BIOMARKERS: Final[list[str]] = [
    "entrohinal_thickness",
    "precuneus_thickness",
    "posteriorcingulate_thickness",
]
VOLUME_BIOMARKERS: Final[list[str]] = [
    "hippocampus_vol",
    "parahippocampal_vol",
    "fusiform_vol",
    "inferiortemporal_vol",
    "ventricular_vol",
]
VOLUME_BIOMARKERS: Final[list[str]] = [
    "hippocampus_vol",
    "parahippocampal_vol",
    "fusiform_vol",
    "inferiortemporal_vol",
    "ventricular_vol",
]
SKEWED_BIOMARKERS: Final[list[str]] = ["ventricular_vol"]
LOWER_IS_ABNORMAL: Final[list[str]] = [
    b for b in EVENT_BIOMARKERS if b != "ventricular_vol"
]
HIGHER_IS_ABNORMAL: Final[list[str]] = ["ventricular_vol"]
VOLUME_RESIDUALIZATION_FORMULA: Final[str] = (
    "{biomarker} ~ age_at_baseline + age_sq + C(sex) + icv"
)
THICKNESS_RESIDUALIZATION_FORMULA: Final[str] = (
    "{biomarker} ~ age_at_baseline + age_sq + C(sex)"
)
N_MCMC_ITER: Final[int] = 50_000
N_BOOTSTRAP: Final[int] = 500
N_PERMUTATION: Final[int] = 5_000
EVENT_SEQUENCE_FILE: Final[Path] = STAGING_RESULTS_DIR / "event_sequence.csv"
SUBJECT_STAGES_FILE: Final[Path] = STAGING_RESULTS_DIR / "subject_stages.csv"
PERMUTATION_TEST_FILE: Final[Path] = STAGING_RESULTS_DIR / "permutation_test.csv"
MIXTURE_MODELS_FILE: Final[Path] = STAGING_RESULTS_DIR / "mixture_models.pkl"
MCMC_SAMPLES_FILE: Final[Path] = STAGING_RESULTS_DIR / "mcmc_samples.npz"
BOOTSTRAP_SEQUENCES_FILE: Final[Path] = STAGING_RESULTS_DIR / "bootstrap_sequences.npz"
OUTCOME_VARIABLES: Final[list[str]] = [
    MMSE_COL,
    CDR_GLOBAL_COL,
]
MIXED_MODEL_FORMULA_TEMPLATE: Final[str] = (
    "{outcome} ~ stage * years_since_baseline + age_at_baseline + C(sex) + education_years" 
)
MIXED_MODEL_FULL_RE_FORMULA: Final[str] = "1 + years_since_baseline"
MIXED_MODEL_FALLBACK_RE_FORMULA: Final[str] = "1"
STAGE_TERTILE_QUANTILES: Final[list[float]] = [0.0, 1/3, 2/3, 1.0]
MMSE_MODEL_RESULTS_FILE: Final[Path] = LONGITUDINAL_RESULTS_DIR / "mmse_model_results.csv"
CDRSUM_MODEL_RESULTS_FILE: Final[Path] = LONGITUDINAL_RESULTS_DIR / "cdrsum_model_results.csv"
PREDICTED_TRAJECTORIES_FILE: Final[Path] = LONGITUDINAL_RESULTS_DIR / "predicted_trajectories.csv"
SEQUENCE_STABILITY_THRESHOLD: Final[float] = 0.8
SENSITIVITY_SUMMARY_FILE: Final[Path] = SENSITIVITY_RESULTS_DIR / "sensitivity_summary.csv"
SENSITIVITY_CHECK_NAMES: Final[list[str]] = [
    "leave_one_biomarker_out",
    "cn_definition_loose",
    "without_age_sq",
    "match_tolerance_365",
]
CENTILOID_VALUE_COL: Final[str] = "Centiloid_fSUVR_rsf_TOT_CORTMEAN"
CENTILOID_TRACER_COL: Final[str] = "tracer"
CENTILOID_SESSION_COL: Final[str] = "oasis_session_id"
AMYLOID_POSITIVITY_CUTOFFS: Final[dict[str, float]] = {
    "AV45": 20.6,
    "PiB": 16.4,
}
BRAAK_SESSION_COL: Final[str] = "OASIS_session_label"
AMYLOID_RESULTS_FILE: Final[Path] = EXTERNAL_VALIDATION_DIR / "amyloid_stage_correlation.csv"
TAU_RESULTS_FILE: Final[Path] = EXTERNAL_VALIDATION_DIR / "tau_stage_correlation.csv"
APOE_STAGE_FILE: Final[Path] = EXTERNAL_VALIDATION_DIR / "apoe_by_stage_tertile.csv"
FIGURE_DPI: Final[int] = 300
OUTPUT_MANIFEST: Final[list[Path]] = [
    FIGURES_DIR / "fig01_event_sequence_pvd.svg",
    FIGURES_DIR / "fig02_mixture_fits.png",
    FIGURES_DIR / "fig03_stage_by_diagnosis.png",
    FIGURES_DIR / "fig04_predicted_trajectories.png",
    FIGURES_DIR / "fig05_forest_plot.png",
    FIGURES_DIR / "fig06_sensitivity_comparison.png",
    FIGURES_DIR / "fig07_apoe_by_stage.png",
    TABLES_DIR / "table01_cohort_characteristics.csv",
    TABLES_DIR / "table02_biomarker_list.csv",
    TABLES_DIR / "table03_event_sequence.csv",
    TABLES_DIR / "table04_mixed_model_results.csv",
    TABLES_DIR / "table05_sensitivity_summary.csv",
    TABLES_DIR / "table06_adni_comparison.csv",
]
PACKAGE_VERSION_FILE: Final[Path] = DOCS_DIR / "environment_snapshot.txt"