from __future__ import annotations 
import sys
from pathlib import Path
from typing import Any 
import numpy as np
import pandas as pd
 
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
 
import config
from utils.io_utils import (
    build_run_metadata,
    file_fingerprint,
    save_csv,
    save_json,
    write_metadata,
)
from utils.logging_utils import (
    append_exclusion_log,
    get_logger,
    log_cascading_filter,
    log_dataframe_shape,
    log_warning_file,
)
from utils.validation_utils import (
    ValidationError,
    assert_columns_present,
    assert_no_duplicate_id,
    assert_non_empty_dataframe,
    assert_value_within_range,
)
 
logger = get_logger(__name__, log_file=config.LOGS_DIR / "02_feature_engineering.log")

EBM_MRI_PATH = config.PROCESSED_DIR / "ebm_mri_dataset.csv"
EBM_AMYLOID_PATH = config.PROCESSED_DIR / "ebm_mri_amyloid_dataset.csv"
EBM_TAU_PATH = config.PROCESSED_DIR / "ebm_mri_amyloid_tau_dataset.csv"
LONGITUDINAL_PATH  = config.PROCESSED_DIR / "longitudinal_validation_dataset.csv"
FE_SUMMARY_PATH  = config.PROCESSED_DIR / "feature_engineering_summary.csv"
FE_METADATA_PATH  = config.PROCESSED_DIR / "feature_engineering_metadata.json"

MRI_PRIMARY_Z_COLS: list[str] = [
    "z_hippocampus_vol",
    "z_entorhinal_thickness",
    "z_fusiform_vol",
    "z_inferiortemporal_vol",
    "z_ventricular_vol",
]


MRI_SENSITIVITY_Z_COLS: str = "z_whole_brain_vol"
AMYLOID_Z_COL: str = "z_Centiloid_fSUVR_TOT_CORTMEAN"
TAU_Z_COL: str = "z_Tauopathy"
MRI_AMYLOID_Z_COLS: list[str] = MRI_PRIMARY_Z_COLS + [AMYLOID_Z_COL]
MRI_AMYLOID_TAU_Z_COLS: list[str] = MRI_PRIMARY_Z_COLS + [AMYLOID_Z_COL, TAU_Z_COL]

EBM_META_COLS: list[str] = [
    config.ID_COL,
    "diagnosis_group",
    "age_at_baseline",
    "sex",
    "education_years",
    "apoe4_carrier",
    "apoe_genotype",
    config.MMSE_COL,
    config.CDR_GLOBAL_COL,
    config.CDR_SB_COL,
    "adequate_followup",
    "n_followup_visits",
    "hippocampus_vol",
    "entorhinal_thickness",
    "fusiform_vol",
    "inferiortemporal_vol",
    "ventricular_vol",
    "whole_brain_vol",
    "Centiloid_fSUVR_TOT_CORTMEAN",
    "Tauopathy",
    "icv",
    "mri_days",
]

LONGITUDINAL_COLS: list[str] = [
    config.ID_COL,
    "diagnosis_group",
    "adequate_followup",
    "n_followup_visits",
    "age_at_baseline",
    "sex",
    "education_years",
    "apoe4_carrier",
    "apoe_genotype",
    config.MMSE_COL,
    config.CDR_GLOBAL_COL,
    config.CDR_SB_COL,
    "mri_days",
    *MRI_PRIMARY_Z_COLS,
    MRI_SENSITIVITY_Z_COLS,
    AMYLOID_Z_COL,
    TAU_Z_COL,
]

_MIN_EBM_NON_CN: int = config.MIN_N_FOR_EBM
_MIN_LONGITUDINAL: int = config.MIN_N_FOR_LONGITUDINAL

def load_analysis_dataset() -> pd.DataFrame:
    """Load analysis_dataset.csv and run structural validation.
 
    Returns
    -------
    pd.DataFrame  One row per subject, 1163 subjects expected.
 
    Raises
    ------
    FileNotFoundError  If the file does not exist.
    ValidationError    If required columns are absent or IDs are duplicated.
    """
    path = config.PROCESSED_DIR / "analysis_dataset.csv"
    if not path.exists():
        raise FileNotFoundError(
            f"analysis_dataset.csv not found at {path}. "
            "Run 01_data_prep.py first."
        )
 
    df = pd.read_csv(path, low_memory=False)
    log_dataframe_shape(logger, df, "analysis_dataset")
 
    required = (
        [config.ID_COL, "ebm_eligible", "diagnosis_group",
         "adequate_followup", "n_followup_visits"]
        + MRI_PRIMARY_Z_COLS
        + [MRI_SENSITIVITY_Z_COLS]
    )
    assert_columns_present(df, required)
    assert_no_duplicate_id(df, config.ID_COL)
    assert_non_empty_dataframe(df, "analysis_dataset")
 
    return df
 
 
def load_longitudinal_raw() -> pd.DataFrame | None:
    """Load longitudinal_raw.csv to compute per-subject follow-up years.
 
    Returns None if the file is not present (follow-up years set to NaN).
    """
    path = config.INTERIM_DIR / "longitudinal_raw.csv"
    if not path.exists():
        logger.warning(
            "longitudinal_raw.csv not found at %s. "
            "'followup_years' will be NaN in the longitudinal dataset.", path,
        )
        return None
 
    long = pd.read_csv(path, low_memory=False)
    log_dataframe_shape(logger, long, "longitudinal_raw")
    return long
 
def _safe_meta_cols(df: pd.DataFrame, extra_z_cols: list[str]) -> list[str]:
    """Return the union of EBM_META_COLS and *extra_z_cols* that are present
    in *df*, preserving order and logging any absent optional columns."""
    all_requested = list(dict.fromkeys(EBM_META_COLS + extra_z_cols))
    present  = [c for c in all_requested if c in df.columns]
    absent = [c for c in all_requested if c not in df.columns]
    if absent:
        logger.info(
            "Optional metadata column(s) not in analysis_dataset — skipped: %s", absent
        )
    return present
 
 
def build_ebm_panel(
    df: pd.DataFrame,
    panel_z_cols: list[str],
    panel_name: str,
    extra_z_cols: list[str] | None = None,
    exclusion_log: list[dict[str, Any]] | None = None,
) -> pd.DataFrame:
    """Build a complete-case EBM dataset for a given biomarker panel.
 
    Steps
    -----
    1. Restrict to EBM-eligible subjects (ebm_eligible == 1).
    2. Retain only subjects with non-null values in all *panel_z_cols*.
    3. Keep EBM_META_COLS + *panel_z_cols* + *extra_z_cols*.
    4. Validate and log.
 
    Args
    ----
    df : Full analysis_dataset dataframe.
    panel_z_cols  : Primary z-score columns that must all be non-null.
    panel_name : Short label used in log messages and the summary table.
    extra_z_cols  : Additional z-score columns to include as optional
                    columns (not used in the completeness filter).
    exclusion_log : In-memory exclusion log list (mutated in-place).
 
    Returns
    -------
    pd.DataFrame  Complete-case EBM dataset, one row per subject.
 
    Raises
    ------
    ValidationError  If the resulting dataset is empty or too small.
    """
    if exclusion_log is None:
        exclusion_log = []
 
    n_total = len(df)
    eligible = df[df["ebm_eligible"].eq(1)].copy()
    n_eligible = len(eligible)
    append_exclusion_log(exclusion_log, f"{panel_name}_ebm_eligibility",
                         n_total, n_eligible)
    log_cascading_filter(logger, f"{panel_name}_ebm_eligibility",
                         n_total, n_eligible)
    assert_columns_present(eligible, panel_z_cols)
    complete = eligible.dropna(subset=panel_z_cols).copy()
    n_complete = len(complete)
    append_exclusion_log(exclusion_log, f"{panel_name}_complete_case",
                         n_eligible, n_complete)
    log_cascading_filter(logger, f"{panel_name}_complete_case",
                         n_eligible, n_complete)
 
    if n_complete == 0:
        raise ValidationError(
            f"Panel '{panel_name}': no subjects remain after complete-case "
            f"filter on {panel_z_cols}."
        )
    
    n_non_cn = int((complete["diagnosis_group"] != config.DX_CN).sum())
    if n_non_cn < _MIN_EBM_NON_CN:
        msg = (
            f"Panel '{panel_name}': only {n_non_cn} non-CN subjects "
            f"(minimum recommended: {_MIN_EBM_NON_CN}). "
            "Treat EBM results as exploratory."
        )
        log_warning_file(msg, config.COHORT_SIZE_WARNING_FILE)
        logger.warning(msg)

    extra = list(extra_z_cols) if extra_z_cols else []
    keep  = _safe_meta_cols(complete, panel_z_cols + extra)
    panel_df = complete[keep].reset_index(drop=True)
    assert_no_duplicate_id(panel_df, config.ID_COL)
    assert_non_empty_dataframe(panel_df, panel_name)

    for z_col in panel_z_cols:
        n_null = int(panel_df[z_col].isna().sum())
        if n_null:
            raise ValidationError(
                f"Panel '{panel_name}': {n_null} null value(s) in '{z_col}' "
                "after complete-case filter — unexpected."
            )
 
    logger.info(
        "Panel '%s': %d subjects  (CN=%d, CIND=%d, AD=%d)",
        panel_name, n_complete,
        int((complete["diagnosis_group"] == config.DX_CN).sum()),
        int((complete["diagnosis_group"] == config.DX_CIND).sum()),
        int((complete["diagnosis_group"] == config.DX_AD).sum()),
    )
 
    return panel_df

def compute_followup_years(
    df: pd.DataFrame,
    longitudinal: pd.DataFrame | None,
) -> pd.DataFrame:
    """Add 'followup_years' — maximum follow-up duration per subject.
 
    If *longitudinal* is available, followup_years = maximum
    years_since_baseline across all qualifying post-baseline visits.
    Otherwise the column is NaN.
 
    Args
    ----
    df          : EBM-eligible analysis_dataset rows.
    longitudinal: Long-format visit table or None.
 
    Returns
    -------
    df copy with 'followup_years' column added.
    """
    df = df.copy()
 
    if longitudinal is None or longitudinal.empty:
        df["followup_years"] = np.nan
        return df
 
    if "years_since_baseline" not in longitudinal.columns:
        logger.warning(
            "'years_since_baseline' not in longitudinal_raw.csv; "
            "'followup_years' set to NaN."
        )
        df["followup_years"] = np.nan
        return df
 
    min_years = config.MIN_FOLLOWUP_DAYS / 365.25
    qualifying = longitudinal[longitudinal["years_since_baseline"] >= min_years]
 
    max_years = (
        qualifying.groupby(config.ID_COL)["years_since_baseline"]
        .max()
        .rename("followup_years")
    )
 
    df["followup_years"] = df[config.ID_COL].map(max_years)
    n_missing = int(df["followup_years"].isna().sum())
    if n_missing:
        logger.info(
            "%d subjects have no qualifying follow-up visits; "
            "'followup_years' is NaN for them.", n_missing,
        )
 
    return df
 
 
def build_longitudinal_dataset(
    df: pd.DataFrame,
    longitudinal: pd.DataFrame | None,
) -> pd.DataFrame:
    """Build the longitudinal validation dataset.
 
    Includes all EBM-eligible subjects regardless of follow-up status.
    The 'adequate_followup' flag allows downstream scripts to restrict to
    subjects with sufficient longitudinal data.
 
    An 'ebm_stage' placeholder column (NaN) is included so that
    03_ebm_staging.py can merge stage assignments without schema changes.
 
    Args
    ----
    df          : Full analysis_dataset dataframe.
    longitudinal: Long-format visit table or None.
 
    Returns
    -------
    pd.DataFrame  One row per subject, all EBM-eligible.
    """
    eligible = df[df["ebm_eligible"].eq(1)].copy()
    n_eligible = len(eligible)
 
    eligible = compute_followup_years(eligible, longitudinal)

    all_requested = list(dict.fromkeys(LONGITUDINAL_COLS + ["followup_years"]))
    keep = [c for c in all_requested if c in eligible.columns]
    missing_requested = [c for c in all_requested if c not in eligible.columns]
    if missing_requested:
        logger.info(
            "Longitudinal dataset: requested column(s) not found, skipped: %s",
            missing_requested,
        )
 
    long_df = eligible[keep].copy()
    long_df["ebm_stage_mri"]  = np.nan
    long_df["ebm_stage_mri_amyloid"] = np.nan
    long_df["ebm_stage_mri_tau"]  = np.nan
 
    long_df = long_df.reset_index(drop=True)
 
    assert_no_duplicate_id(long_df, config.ID_COL)
    assert_non_empty_dataframe(long_df, "longitudinal_validation_dataset")
 
    n_adequate = int(long_df["adequate_followup"].sum())
    logger.info(
        "Longitudinal dataset: %d subjects  (%d with adequate follow-up, "
        "%.1f%%)",
        n_eligible, n_adequate,
        100.0 * n_adequate / n_eligible if n_eligible else 0.0,
    )
 
    if n_adequate < _MIN_LONGITUDINAL:
        msg = (
            f"Only {n_adequate} EBM-eligible subjects have adequate follow-up "
            f"(minimum recommended: {_MIN_LONGITUDINAL})."
        )
        log_warning_file(msg, config.COHORT_SIZE_WARNING_FILE)
        logger.warning(msg)
 
    return long_df

def _diagnosis_counts(df: pd.DataFrame, label: str) -> list[dict[str, Any]]:
    """Return per-diagnosis-group counts as a list of dicts."""
    dx_col = "diagnosis_group"
    rows = []
    for dx in [config.DX_CN, config.DX_CIND, config.DX_AD,
                config.DX_OTHER, config.DX_UNCLASSIFIED]:
        n = int((df[dx_col] == dx).sum()) if dx_col in df.columns else 0
        rows.append({
            "dataset": label,
            "category": "diagnosis",
            "subcategory": dx,
            "n": n,
            "pct_of_panel": round(100.0 * n / len(df), 1) if len(df) else 0.0,
        })
    return rows
 
 
def _z_score_stats(df: pd.DataFrame, z_cols: list[str], label: str) -> list[dict[str, Any]]:
    """Return per-biomarker descriptive statistics (mean, sd, min, max, n_null)."""
    rows = []
    for col in z_cols:
        if col not in df.columns:
            continue
        s = df[col]
        rows.append({
            "dataset": label,
            "category":  "biomarker_stats",
            "biomarker": col,
            "n": int(s.notna().sum()),
            "n_null": int(s.isna().sum()),
            "mean": round(float(s.mean()), 4),
            "sd": round(float(s.std(ddof=1)), 4),
            "min": round(float(s.min()), 4),
            "p25": round(float(s.quantile(0.25)), 4),
            "median": round(float(s.median()), 4),
            "p75": round(float(s.quantile(0.75)), 4),
            "max": round(float(s.max()), 4),
        })
    return rows
 
 
def _missingness_row(df: pd.DataFrame, col: str, label: str) -> dict[str, Any]:
    n_null = int(df[col].isna().sum()) if col in df.columns else len(df)
    return {
        "dataset":  label,
        "category": "missingness",
        "column": col,
        "n_null": n_null,
        "pct_null": round(100.0 * n_null / len(df), 2) if len(df) else 0.0,
    }
 
 
def build_summary(
    mri_df: pd.DataFrame,
    amyloid_df: pd.DataFrame,
    tau_df: pd.DataFrame,
    long_df: pd.DataFrame,
) -> pd.DataFrame:
    """Assemble a comprehensive QC summary table across all four datasets.
 
    Returns
    -------
    pd.DataFrame with columns [dataset, category, subcategory/biomarker/column,
    n, pct_of_panel, mean, sd, min, p25, median, p75, max, n_null, pct_null].
    """
    rows: list[dict[str, Any]] = []

    for label, panel in [
        ("mri_only", mri_df),
        ("mri_amyloid", amyloid_df),
        ("mri_amyloid_tau", tau_df),
        ("longitudinal", long_df),
    ]:
        rows.append({
            "dataset": label, "category": "sample_size",
            "subcategory": "n_subjects", "n": len(panel),
            "pct_of_panel": 100.0,
        })
    
    for label, panel in [
        ("mri_only", mri_df),
        ("mri_amyloid", amyloid_df),
        ("mri_amyloid_tau", tau_df),
        ("longitudinal", long_df),
    ]:
        rows.extend(_diagnosis_counts(panel, label))
    
    for label, panel in [("mri_only", mri_df), ("longitudinal", long_df)]:
        if "adequate_followup" in panel.columns:
            n_ok = int(panel["adequate_followup"].sum())
            rows.append({
                "dataset": label, "category": "followup",
                "subcategory": "adequate_followup", "n": n_ok,
                "pct_of_panel": round(100.0 * n_ok / len(panel), 1) if len(panel) else 0.0,
            })
    
    rows.extend(_z_score_stats(mri_df, MRI_PRIMARY_Z_COLS, "mri_only"))
    rows.extend(_z_score_stats(amyloid_df, MRI_AMYLOID_Z_COLS, "mri_amyloid"))
    rows.extend(_z_score_stats(tau_df, MRI_AMYLOID_TAU_Z_COLS, "mri_amyloid_tau"))

    all_z = list(dict.fromkeys(
        MRI_PRIMARY_Z_COLS + [MRI_SENSITIVITY_Z_COLS] + [AMYLOID_Z_COL, TAU_Z_COL]
    ))
    for label, panel in [
        ("mri_only", mri_df),
        ("mri_amyloid", amyloid_df),
        ("mri_amyloid_tau", tau_df),
        ("longitudinal", long_df),
    ]:
        for col in all_z:
            rows.append(_missingness_row(panel, col, label))
 
    summary = pd.DataFrame(rows)
    for col in ["mean", "sd", "min", "p25", "median", "p75", "max",
                "n_null", "pct_null"]:
        if col not in summary.columns:
            summary[col] = np.nan
    return summary

def save_outputs(
    mri_df: pd.DataFrame,
    amyloid_df: pd.DataFrame,
    tau_df: pd.DataFrame,
    long_df: pd.DataFrame,
    summary: pd.DataFrame,
    exclusion_log: list[dict[str, Any]],
) -> None:
    """Write all outputs and provenance metadata.
 
    Files written
    -------------
    data/processed/ebm_mri_dataset.csv
    data/processed/ebm_mri_amyloid_dataset.csv
    data/processed/ebm_mri_amyloid_tau_dataset.csv
    data/processed/longitudinal_validation_dataset.csv
    data/processed/feature_engineering_summary.csv
    data/processed/feature_engineering_metadata.json
    """
    config.ensure_project_dirs()
 
    save_csv(mri_df,     EBM_MRI_PATH)
    save_csv(amyloid_df, EBM_AMYLOID_PATH)
    save_csv(tau_df,     EBM_TAU_PATH)
    save_csv(long_df,    LONGITUDINAL_PATH)
    save_csv(summary,    FE_SUMMARY_PATH)
 
    logger.info("Saved: %s (%d x %d)", EBM_MRI_PATH.name, *mri_df.shape)
    logger.info("Saved: %s (%d x %d)", EBM_AMYLOID_PATH.name, *amyloid_df.shape)
    logger.info("Saved: %s (%d x %d)", EBM_TAU_PATH.name, *tau_df.shape)
    logger.info("Saved: %s (%d x %d)", LONGITUDINAL_PATH.name, *long_df.shape)
    logger.info("Saved: %s (%d x %d)", FE_SUMMARY_PATH.name, *summary.shape)

    source_path = config.PROCESSED_DIR / "analysis_dataset.csv"
    meta = build_run_metadata("02_feature_engineering.py")
    meta.update({
        "project_version": config.PROJECT_VERSION,
        "source_fingerprint": (
            file_fingerprint(source_path) if source_path.exists() else "MISSING"
        ),
        "panels": {
            "mri_only": {
                "biomarkers": MRI_PRIMARY_Z_COLS,
                "sensitivity_extra": [MRI_SENSITIVITY_Z_COLS],
                "n_subjects": len(mri_df),
                "path": str(EBM_MRI_PATH),
            },
            "mri_amyloid": {
                "biomarkers": MRI_AMYLOID_Z_COLS,
                "n_subjects": len(amyloid_df),
                "path": str(EBM_AMYLOID_PATH),
            },
            "mri_amyloid_tau": {
                "biomarkers": MRI_AMYLOID_TAU_Z_COLS,
                "n_subjects": len(tau_df),
                "path": str(EBM_TAU_PATH),
            },
            "longitudinal": {
                "n_subjects": len(long_df),
                "n_adequate_followup": int(long_df["adequate_followup"].sum()),
                "path": str(LONGITUDINAL_PATH),
                "ebm_stage_cols": ["ebm_stage_mri", "ebm_stage_mri_amyloid", "ebm_stage_mri_tau"],
                "note": (
                    "ebm_stage_* columns are NaN placeholders; "
                    "filled by 03_ebm_staging.py"
                ),
            },
        },
        "exclusion_log": exclusion_log,
    })
 
    write_metadata(meta, FE_METADATA_PATH)
    logger.info("Saved:  %s", FE_METADATA_PATH.name)

def main() -> None:
    """Orchestrate the feature-engineering pipeline."""
    config.ensure_project_dirs()
 
    logger.info("=" * 70)
    logger.info("02_feature_engineering.py  |  version %s", config.PROJECT_VERSION)
    logger.info("=" * 70)
 
    exclusion_log: list[dict[str, Any]] = []

    df           = load_analysis_dataset()
    longitudinal = load_longitudinal_raw()
 
    n_total   = len(df)
    n_eligible = int(df["ebm_eligible"].sum())
    logger.info(
        "Input: %d total subjects  |  %d EBM-eligible", n_total, n_eligible
    )

    logger.info("--- Building MRI-only panel ---")
    mri_df = build_ebm_panel(
        df,
        panel_z_cols   = MRI_PRIMARY_Z_COLS,
        panel_name     = "mri_only",
        extra_z_cols   = [MRI_SENSITIVITY_Z_COLS],
        exclusion_log  = exclusion_log,
    )
 
    logger.info("--- Building MRI + Amyloid panel ---")
    amyloid_df = build_ebm_panel(
        df,
        panel_z_cols  = MRI_AMYLOID_Z_COLS,
        panel_name    = "mri_amyloid",
        exclusion_log = exclusion_log,
    )
 
    logger.info("--- Building MRI + Amyloid + Tau panel ---")
    tau_df = build_ebm_panel(
        df,
        panel_z_cols  = MRI_AMYLOID_TAU_Z_COLS,
        panel_name    = "mri_amyloid_tau",
        exclusion_log = exclusion_log,
    )

    logger.info("--- Building longitudinal validation dataset ---")
    long_df = build_longitudinal_dataset(df, longitudinal)
    logger.info("--- Building QC summary ---")
    summary = build_summary(mri_df, amyloid_df, tau_df, long_df)
 
    save_outputs(mri_df, amyloid_df, tau_df, long_df, summary, exclusion_log)
    logger.info("=" * 70)
    logger.info("FEATURE ENGINEERING COMPLETE")
    logger.info("  MRI-only panel:            %d subjects  (%d cols)",
                len(mri_df), mri_df.shape[1])
    logger.info("  MRI + Amyloid panel:       %d subjects  (%d cols)",
                len(amyloid_df), amyloid_df.shape[1])
    logger.info("  MRI + Amyloid + Tau panel: %d subjects  (%d cols)",
                len(tau_df), tau_df.shape[1])
    logger.info("  Longitudinal dataset:      %d subjects  (%d cols)",
                len(long_df), long_df.shape[1])
    logger.info("=" * 70)
 
 
if __name__ == "__main__":
    main()
 


