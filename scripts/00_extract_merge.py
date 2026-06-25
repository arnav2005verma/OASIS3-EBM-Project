from __future__ import annotations
import sys
from pathlib import Path
import pandas as pd
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
import config
from utils.io_utils import (
    build_run_metadata,
    save_csv,
    save_environment_snapshot,
    write_metadata,
)
from utils.logging_utils import (
    append_exclusion_log,
    get_logger,
    log_cascading_filter,
    log_dataframe_shape,
)
from utils.validation_utils import (
    ValidationError,
    assert_columns_present,
    assert_no_duplicate_id,
    assert_value_within_range,
)
logger = get_logger(__name__, log_file=config.LOGS_DIR / "00_extract_merge.log")
_FS_REQUIRED = [
    config.FREESURFER_SUBJECT_COL,
    config.FREESURFER_SESSION_COL,
    config.FS_QC_STATUS_COL,
    config.FS_VERSION_COL,
    config.FS_ICV_COL,
    config.FS_HIPPOCAMPUS_TOT_COL,
    config.FS_ENTORHINAL_LH_COL,
    config.FS_ENTORHINAL_RH_COL,
    config.FS_PARAHIPPOCAMPAL_LH_COL,
    config.FS_PARAHIPPOCAMPAL_RH_COL,
    config.FS_FUSIFORM_LH_COL,
    config.FS_FUSIFORM_RH_COL,
    config.FS_INFERIORTEMPORAL_LH_COL,
    config.FS_INFERIORTEMPORAL_RH_COL,
    config.FS_PRECUNEUS_LH_COL,
    config.FS_PRECUNEUS_RH_COL,
    config.FS_POSTERIORCINGULATE_LH_COL,
    config.FS_POSTERIORCINGULATE_RH_COL,
    *config.VENTRICULAR_VOL_COMPONENTS,
    *config.WHOLE_BRAIN_VOL_COMPONENTS,
]
 
_D1_REQUIRED = [
    config.ID_COL,
    config.UDS_DAYS_COL,
    config.NORMCOG_COL,
    config.DEMENTED_COL,
    config.PROBAD_COL,
    config.POSSAD_COL,
    *config.MCI_FLAG_COLS,
    *config.NON_AD_EXCLUSION_COLS,
]
 
_B4_REQUIRED = [
    config.ID_COL,
    config.UDS_DAYS_COL,
    config.MMSE_COL,
    config.CDR_GLOBAL_COL,
    config.CDR_SB_COL,
]
 
_DEMO_REQUIRED = [
    config.ID_COL,
    config.DEMO_AGE_AT_ENTRY_COL,
    config.DEMO_SEX_COL,
    config.DEMO_EDUCATION_COL,
    config.DEMO_APOE_COL,
]
 
_CENTILOID_REQUIRED = [
    "subject_id",
    config.CENTILOID_SESSION_COL,
    config.CENTILOID_TRACER_COL,
    config.CENTILOID_VALUE_COL,
]
 
_BRAAK_REQUIRED = [
    config.BRAAK_ID_COL,
    config.BRAAK_SESSION_COL,
    config.UDS_DAYS_COL,
    config.TAUPATHY_COL,
]
def _read_csv_loud(key: str) -> pd.DataFrame:
    """Read a raw CSV from config.RAW_FILE_PATHS[key], raising if missing."""
    path = config.RAW_FILE_PATHS[key]
    if not path.exists():
        raise FileNotFoundError(
            f"RAW file '{key} not found at {path}. "
            "Ensure OASIS-3 exports have been place in data/raw/ and that "
            "config.RAW_FILE_PATHS points to the correct filenames"
        )
    df = pd.read_csv(path, low_memory=False)
    logger.info("Loaded %-14s -> %d rows x %d cols", key, *df.shape)
    return df 
def load_freesurfer() -> pd.DataFrame: 
    """Load OASIS3_Freesurfer_output.csv and standarise the subject ID column/

    Returns a dataframe with OASISID as the subject identifier. The raw 'Subject' column is retained as 'fs_subject_raw' for traceability.
    """
    df = _read_csv_loud("freesurfer")
    assert_columns_present(df, _FS_REQUIRED)
    df = df.rename(columns={config.FREESURFER_SUBJECT_COL: config.ID_COL})
    hyphen_rename = {c: c.replace("-", "_") for c in config.VENTRICULAR_VOL_COMPONENTS}
    df = df.rename(columns=hyphen_rename)
    log_dataframe_shape(logger, df, "Freesurffer (raw)")
    return df 
def load_udsd1() -> pd.DataFrame: 
    """Load OASIS3_UDSd1_diagnosis.csv.

    The UDSd1 file already uses 'OASISID' as the subject column. 
    Retruns a subset of columns required by the pipeline.
    """
    df = _read_csv_loud("udsd1")
    assert_columns_present(df, _D1_REQUIRED)
    keep = list({config.ID_COL, config.UDS_DAYS_COL, config.UDS_SESSION_LABEL_COL,
                 config.NORMCOG_COL, config.DEMENTED_COL, config.PROBAD_COL,
                 config.POSSAD_COL, *config.MCI_FLAG_COLS, *config.NON_AD_EXCLUSION_COLS}
                 & set(df.columns))
    df = df[keep].copy()
    log_dataframe_shape(logger, df, "UDSd1")
    return df
def load_udsb4() -> pd.DataFrame:
    """Load OASIS3_UDSb4_cdr.csv.
 
    Returns a subset of columns required by the pipeline (MMSE, CDRTOT, CDRSUM).
    """
    df = _read_csv_loud("udsb4")
    assert_columns_present(df, _B4_REQUIRED)
    keep = [config.ID_COL, config.UDS_DAYS_COL, config.UDS_SESSION_LABEL_COL,
            config.MMSE_COL, config.CDR_GLOBAL_COL, config.CDR_SB_COL]
    keep = [c for c in keep if c in df.columns]
    df = df[keep].copy()
    log_dataframe_shape(logger, df, "UDSb4")
    return df
 
 
def load_demographics() -> pd.DataFrame:
    """Load OASIS3_demographics.csv.
 
    Recode GENDER (1/2 int) to sex ('M'/'F') and APOE (two-digit float) to
    apoe4_carrier (0/1 int).  Both GENDER and APOE NaN values are preserved
    rather than silently filled.
    """
    df = _read_csv_loud("demographics")
    assert_columns_present(df, _DEMO_REQUIRED)
 
    df = df.rename(columns={
        config.DEMO_AGE_AT_ENTRY_COL: "age_at_entry",
        config.DEMO_EDUCATION_COL:    "education_years",
    })
    df["sex"] = df[config.DEMO_SEX_COL].map(config.DEMO_SEX_RECODE)
    unmapped_sex = df[config.DEMO_SEX_COL].notna() & df["sex"].isna()
    if unmapped_sex.any():
        logger.warning(
            "%d subjects have unrecognised GENDER values: %s",
            unmapped_sex.sum(),
            df.loc[unmapped_sex, config.DEMO_SEX_COL].unique().tolist(),
        )
    def _recode_apoe4(val: float) -> int | float:
        if pd.isna(val):
            return np.nan
        return int(int(val) in config.APOE_E4_GENOTYPES)
    df["apoe4_carrier"] = df[config.DEMO_APOE_COL].apply(_recode_apoe4)
    df["apoe_genotype"] = df[config.DEMO_APOE_COL].apply(
        lambda v: str(int(v)) if pd.notna(v) else np.nan
    )
    assert_no_duplicate_id(df, config.ID_COL)
    log_dataframe_shape(logger, df, "Demographics")
    return df
def load_centiloid() -> pd.DataFrame:
    """Load OASIS3_amyloid_centiloid.csv and standardise the subject ID.
 
    The centiloid file uses 'subject_id' rather than 'OASISID'.  This
    function renames it immediately.  The returned dataframe is used only
    for logging/validation in this script; the external-validation script
    (05b) loads it directly from RAW_DIR.
    """
    df = _read_csv_loud("centiloid")
    assert_columns_present(df, _CENTILOID_REQUIRED)
    df = df.rename(columns={"subject_id": config.ID_COL})
    log_dataframe_shape(logger, df, "Centiloid")
    return df
 
 
def load_braak() -> pd.DataFrame:
    """Load OASIS3_AV1451_braak_tauopathy.csv and standardise the subject ID.
 
    The Braak file uses 'OASIS_ID' rather than 'OASISID'.
    """
    df = _read_csv_loud("braak")
    assert_columns_present(df, _BRAAK_REQUIRED)
    df = df.rename(columns={config.BRAAK_ID_COL: config.ID_COL})
    log_dataframe_shape(logger, df, "Braak")
    return df
def parse_session_days(df: pd.DataFramw)  -> pd.DataFrame:
    """Extract integer days-from-entry from the MR_session label.
 
    Session labels follow the format 'OAS3XXXX_MR_dNNNN' where NNNN is the
    days-from-entry integer.  Malformed labels (those that do not end with a
    parseable non-negative integer after '_d') raise ValueError.
 
    Adds a 'mri_days' column to the dataframe.  The original session column
    is preserved.
 
    Args:
        df: FreeSurfer dataframe containing config.FREESURFER_SESSION_COL.
 
    Returns:
        A copy of df with 'mri_days' (int) added.
 
    Raises:
        ValueError: If any session label cannot be parsed.
    """
    df = df.copy()
    raw_days = df[config.FREESURFER_SESSION_COL].str.split("_d").str[-1]
    bad_mask = ~raw_days.str.match(r"^\d+$", na=False) | raw_days.isna()
    if bad_mask.any():
        bad_labels = df.loc[bad_mask, config.FREESURFER_SESSION_COL].tolist()
        raise ValueError(
            f"Could not parse days from {bad_mask.sum()} MR_session label(s). "
            f"Expected format '..._dNNNN'. Offending labels: {bad_labels[:5]}"
        )
    df["mri_days"] = raw_days.astype(int)
    return df
def apply_fs_filters(df: pd.DataFrame) -> tuple[pd.DataFrame, set[str]]:
    """Apply FreeSurfer version and QC status filters.
 
    Version filter: sessions whose 'version' field contains
    config.FS_VERSION_FILTER (case-sensitive substring match).
 
    QC filter: sessions whose 'FS QC Status' matches one of
    config.FS_QC_PASS_VALUES (case-insensitive).
 
    Args:
        df: Raw FreeSurfer dataframe with OASISID already renamed.
 
    Returns:
        (filtered_df, excluded_ids) where excluded_ids is the set of subject
        IDs that had NO remaining session after filtering.  Subjects who had
        at least one session survive; they will be thinned further by
        baseline-selection.
    """
    n_sessions_in = len(df)
 
    version_mask = df[config.FS_VERSION_COL].str.contains(
        config.FS_VERSION_FILTER, na=False, regex=False
    )
    df_v = df[version_mask].copy()
    log_cascading_filter(
        logger, "fs_version",
        n_sessions_in, len(df_v),
    )
 
    qc_mask = df_v[config.FS_QC_STATUS_COL].str.lower().isin(config.FS_QC_PASS_VALUES)
    df_qc = df_v[qc_mask].copy()
    log_cascading_filter(
        logger, "fs_qc_status",
        len(df_v), len(df_qc),
    )
    subjects_before = set(df[config.ID_COL].unique())
    subjects_after  = set(df_qc[config.ID_COL].unique())
    excluded_ids = subjects_before - subjects_after
    if excluded_ids:
        logger.info(
            "%d subjects have no sessions surviving QC/version filters.",
            len(excluded_ids),
        )
    return df_qc, excluded_ids
def compute_derived_biomarkers(df: pd.DataFrame) -> pd.DataFrame:
    """Compute pipeline-internal biomarker columns from FreeSurfer source columns.
 
    Bilateral averages are used for thickness biomarkers; bilateral sums for
    volume biomarkers.  ICV correction is NOT applied here — that belongs in
    02_feature_engineering.py.
 
    The original source columns are preserved alongside the derived columns.
 
    Args:
        df: FreeSurfer dataframe with hyphenated column names already renamed
            to underscores (performed in load_freesurfer).
 
    Returns:
        df with the following columns added:
          hippocampus_vol, entorhinal_thickness, parahippocampal_vol,
          fusiform_vol, inferiortemporal_vol, precuneus_thickness,
          posteriorcingulate_thickness, ventricular_vol,
          whole_brain_vol (descriptive only, not an EBM event),
          icv (alias for IntraCranialVol).
    """
    df = df.copy()
    df["hippocampus_vol"] = df[config.FS_HIPPOCAMPUS_TOT_COL]
    df["entrohinal_thickness"] = (
        df[config.FS_ENTORHINAL_LH_COL] + df[config.FS_ENTORHINAL_RH_COL]
    ) / 2.0
    df["precuneus_thickness"] = (
        df[config.FS_PRECUNEUS_LH_COL] + df[config.FS_PRECUNEUS_RH_COL]
    ) / 2.0
 
    df["posteriorcingulate_thickness"] = (
        df[config.FS_POSTERIORCINGULATE_LH_COL] + df[config.FS_POSTERIORCINGULATE_RH_COL]
    ) / 2.0
    df["parahippocampal_vol"] = (
        df[config.FS_PARAHIPPOCAMPAL_LH_COL] + df[config.FS_PARAHIPPOCAMPAL_RH_COL]
    )
 
    df["fusiform_vol"] = (
        df[config.FS_FUSIFORM_LH_COL] + df[config.FS_FUSIFORM_RH_COL]
    )
 
    df["inferiortemporal_vol"] = (
        df[config.FS_INFERIORTEMPORAL_LH_COL] + df[config.FS_INFERIORTEMPORAL_RH_COL]
    )
    renamed_vent = [c.replace("-", "_") for c in config.VENTRICULAR_VOL_COMPONENTS]
    df["ventricular_vol"] = df[renamed_vent].sum(axis=1)
    df["whole_brain_vol"] = df[config.WHOLE_BRAIN_VOL_COMPONENTS].sum(axis=1)
    df["icv"] = df[config.FS_ICV_COL]
    return df 
def build_combined_clinical(udsd1: pd.DataFrame, udsb4: pd.DataFrame) -> pd.DataFrame:
    """Merge UDSd1 and UDSb4 on (OASISID, days_to_visit).
 
    An outer join is used so that rows present in only one file are retained
    with NaN for the absent columns.  For baseline matching, only rows where
    both NORMCOG (from UDSd1) and MMSE (from UDSb4) are non-null are
    considered valid baselines.
 
    Args:
        udsd1: UDSd1 dataframe (diagnosis flags).
        udsb4: UDSb4 dataframe (MMSE, CDRTOT, CDRSUM).
 
    Returns:
        One row per (OASISID, days_to_visit) pair.  A boolean column
        'has_d1_data' and 'has_b4_data' flag which source contributed.
    """
    join_cols = [config.ID_COL, config.UDS_DAYS_COL]
 
    combined = pd.merge(
        udsb4,
        udsd1,
        on=join_cols,
        how="outer",
        suffixes=("_b4", "_d1"),
    )
 
    combined["has_b4_data"] = combined[config.MMSE_COL].notna()
    combined["has_d1_data"] = combined[config.NORMCOG_COL].notna()
 
    log_dataframe_shape(logger, combined, "Combined clinical (outer join)")
    n_both = (combined["has_b4_data"] & combined["has_d1_data"]).sum()
    logger.info(
        "Combined clinical: %d rows with both UDSb4 and UDSd1 data (out of %d total).",
        n_both, len(combined),
    )
    return combined
def _nearest_clinical_row(
    subject_clinical: pd.DataFrame,
    mri_day: int,
    tolerance_days: int,
    require_both: bool = True,
) -> pd.Series | None:
    """Find the single nearest clinical row to `mri_day` within `tolerance_days`.
 
    Tie-break rule (documented): among equally-close candidates, the row with
    the smaller days_to_visit (earlier visit) is chosen.
 
    Args:
        subject_clinical: All clinical rows for one subject.
        mri_day: The MRI session's days-from-entry value.
        tolerance_days: Maximum allowed gap between MRI and clinical visit.
        require_both: If True, candidates must have both UDSd1 and UDSb4 data.
 
    Returns:
        The nearest pd.Series, or None if no candidate is within tolerance.
    """
    clin = subject_clinical.copy()
    clin["_gap"] = (clin[config.UDS_DAYS_COL] - mri_day).abs()
 
    if require_both:
        clin = clin[clin["has_b4_data"] & clin["has_d1_data"]]
 
    within = clin[clin["_gap"] <= tolerance_days]
    if within.empty:
        return None 
    within = within.sort_values(["_gap", config.UDS_DAYS_COL])
    return within.iloc[0]
 
 
def select_baseline_sessions(
    fs_filtered: pd.DataFrame,
    combined_clinical: pd.DataFrame,
    tolerance_days: int,
) -> pd.DataFrame:
    """Select one baseline MRI session per subject, with its matched clinical visit.
 
    For each subject, iterates over their QC-pass 5.3 FS sessions in
    ascending mri_days order and returns the earliest session that has a
    valid matched clinical row (both UDSd1 and UDSb4 data) within
    `tolerance_days`.
 
    Subjects with no qualifying session are logged and excluded from the output.
 
    Args:
        fs_filtered: FreeSurfer dataframe after QC/version filtering and
                     biomarker computation.  Must contain 'mri_days'.
        combined_clinical: Merged UDSd1 + UDSb4 table.
        tolerance_days: Maximum MRI–clinical gap in days.
 
    Returns:
        One row per subject containing all FS columns plus the matched
        clinical columns and 'days_mri_to_clinical'.
    """
    results: list[dict] = []
    unmatched_ids: list[str] = []
    n_subjects_with_fs = fs_filtered[config.ID_COL].nunique()
 
    for oasis_id, subj_fs in fs_filtered.groupby(config.ID_COL):
        subj_clinical = combined_clinical[combined_clinical[config.ID_COL] == oasis_id]
 
        if subj_clinical.empty:
            unmatched_ids.append(str(oasis_id))
            continue
        subj_fs_sorted = subj_fs.sort_values(
            ["mri_days", config.FREESURFER_SESSION_COL]
        )
 
        matched_row: dict | None = None
        for _, fs_row in subj_fs_sorted.iterrows():
            nearest = _nearest_clinical_row(
                subj_clinical,
                mri_day=int(fs_row["mri_days"]),
                tolerance_days=tolerance_days,
                require_both=True,
            )
            if nearest is not None:
                merged = {**fs_row.to_dict(), **nearest.to_dict()}
                merged["days_mri_to_clinical"] = int(abs(
                    nearest[config.UDS_DAYS_COL] - fs_row["mri_days"]
                ))
                merged["clinical_days_to_visit"] = int(nearest[config.UDS_DAYS_COL])
                matched_row = merged
                break
 
        if matched_row is not None:
            results.append(matched_row)
        else:
            unmatched_ids.append(str(oasis_id))
 
    if unmatched_ids:
        logger.warning(
            "%d subjects had no QC-pass 5.3 FS session with a matching "
            "clinical visit within %d days.",
            len(unmatched_ids), tolerance_days,
        )
    matched_df = pd.DataFrame(results)
    log_cascading_filter(
        logger, "clinical_matching",
        n_subjects_with_fs, len(matched_df),
    )
    return matched_df
def merge_demographics(
    baseline: pd.DataFrame,
    demographics: pd.DataFrame,
) -> pd.DataFrame:
    """Left-join demographic covariates onto the baseline dataframe.
 
    Computes age_at_baseline from age_at_entry and mri_days.
 
    Args:
        baseline: One-row-per-subject dataframe from select_baseline_sessions.
        demographics: Loaded and recoded demographics dataframe.
 
    Returns:
        baseline with demographic columns added.  Subjects absent from
        demographics (should not occur in OASIS-3 but logged if they do)
        retain NaN for demographic fields.
    """
    demo_cols = [
        config.ID_COL,
        "age_at_entry",
        "sex",
        "education_years",
        "apoe4_carrier",
        "apoe_genotype",
    ]
    demo_subset = demographics[[c for c in demo_cols if c in demographics.columns]]
 
    merged = pd.merge(baseline, demo_subset, on=config.ID_COL, how="left")
 
    missing_demo = merged["age_at_entry"].isna().sum()
    if missing_demo:
        logger.warning(
            "%d subjects have no demographics row (unexpected for OASIS-3).",
            missing_demo,
        )
    merged["age_at_baseline"] = (
        merged["age_at_entry"] + merged["mri_days"] / 365.25
    )
 
    log_dataframe_shape(logger, merged, "After demographics merge")
    return merged
def build_longitudinal_table(
    udsb4: pd.DataFrame,
    baseline: pd.DataFrame,
) -> pd.DataFrame:
    """Build the long-format longitudinal outcomes table.
 
    Includes all UDSb4 rows for each subject where days_to_visit is strictly
    greater than (baseline MRI days + 0) — i.e., visits after the baseline
    MRI date.  Computes years_since_baseline.
 
    Args:
        udsb4: Full UDSb4 dataframe (all visits, all subjects).
        baseline: One-row-per-subject baseline dataframe containing 'mri_days'.
 
    Returns:
        Long-format dataframe with columns:
          OASISID, days_to_visit, years_since_baseline,
          MMSE, CDRTOT, CDRSUM,
          (OASIS_session_label if present in udsb4).
    """
    baseline_days = baseline[[config.ID_COL, "mri_days"]].copy()
    long = pd.merge(udsb4, baseline_days, on=config.ID_COL, how="inner")
 
    # Post-baseline only.
    long = long[long[config.UDS_DAYS_COL] > long["mri_days"]].copy()
 
    long["years_since_baseline"] = (
        (long[config.UDS_DAYS_COL] - long["mri_days"]) / 365.25
    )
 
    keep = [
        config.ID_COL,
        config.UDS_DAYS_COL,
        "years_since_baseline",
        config.MMSE_COL,
        config.CDR_GLOBAL_COL,
        config.CDR_SB_COL,
    ]
    if config.UDS_SESSION_LABEL_COL in long.columns:
        keep.append(config.UDS_SESSION_LABEL_COL)
 
    long = long[[c for c in keep if c in long.columns]].reset_index(drop=True)
    log_dataframe_shape(logger, long, "Longitudinal table (post-baseline UDSb4 visits)")
    return long

_MERGED_RAW_COLUMNS: list[str] = [
    config.ID_COL,
    config.FREESURFER_SESSION_COL,
    "mri_days",
    "clinical_days_to_visit",
    "days_mri_to_clinical",
    config.FS_VERSION_COL,
    config.FS_QC_STATUS_COL,
    "age_at_baseline",
    "age_at_entry",
    "sex",
    "education_years",
    "apoe4_carrier",
    "apoe_genotype",
    "icv",
    "hippocampus_vol",
    "entorhinal_thickness",
    "parahippocampal_vol",
    "fusiform_vol",
    "inferiortemporal_vol",
    "precuneus_thickness",
    "posteriorcingulate_thickness",
    "ventricular_vol",
    "whole_brain_vol",
    config.MMSE_COL,
    config.CDR_GLOBAL_COL,
    config.CDR_SB_COL,
    config.NORMCOG_COL,
    config.DEMENTED_COL,
    config.PROBAD_COL,
    config.POSSAD_COL,
    *config.MCI_FLAG_COLS,
    *config.NON_AD_EXCLUSION_COLS,
]
 
 
def select_merged_raw_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Reduce the merged dataframe to the canonical merged_raw.csv schema.
 
    Columns listed in _MERGED_RAW_COLUMNS that are absent from df are silently
    skipped (they will appear as missing in downstream validation, surfacing
    the issue explicitly at the 01_data_prep.py stage rather than crashing
    here with a confusing KeyError).
 
    Duplicate column names that can arise from the outer clinical join are
    resolved by dropping the suffixed duplicates before selection.
    """
    suffix_pattern = ("_b4", "_d1")
    drop_cols = [c for c in df.columns if c.endswith(suffix_pattern)]
    if drop_cols:
        df = df.drop(columns=drop_cols)
 
    present = [c for c in _MERGED_RAW_COLUMNS if c in df.columns]
    absent  = [c for c in _MERGED_RAW_COLUMNS if c not in df.columns]
    if absent:
        logger.warning(
            "The following expected columns are absent from the merged "
            "dataframe and will be missing in merged_raw.csv: %s", absent
        )
    return df[present].copy()
def validate_merged_raw(df: pd.DataFrame) -> None:
    """Post-merge validation checks on the baseline output.
 
    Raises ValidationError on any structural failure.
    """
    assert_no_duplicate_id(df, config.ID_COL)
 
    assert_value_within_range(
        df["days_mri_to_clinical"],
        min_value=0,
        max_value=config.MATCH_TOLERANCE_DAYS_PRIMARY,
    )
    null_mri_days = df["mri_days"].isna().sum()
    if null_mri_days:
        raise ValidationError(
            f"{null_mri_days} rows in merged_raw have null mri_days — "
            "indicates a merge error in session day parsing."
        )
    for col in [config.MMSE_COL, config.CDR_GLOBAL_COL, config.CDR_SB_COL,
                config.NORMCOG_COL, "icv", "hippocampus_vol"]:
        n_null = df[col].isna().sum() if col in df.columns else len(df)
        if n_null:
            logger.warning("merged_raw: '%s' has %d null value(s).", col, n_null)
 
    logger.info("validate_merged_raw: PASSED (%d subjects).", len(df))
 
 
def validate_longitudinal(df: pd.DataFrame) -> None:
    """Validation checks on the longitudinal output."""
    neg_years = (df["years_since_baseline"] <= 0).sum()
    if neg_years:
        raise ValidationError(
            f"{neg_years} longitudinal rows have years_since_baseline ≤ 0, "
            "indicating baseline or pre-baseline visits were included."
        )
    logger.info("validate_longitudinal: PASSED (%d rows).", len(df))
def log_conservation(
    n_fs_raw: int,
    n_fs_filtered: int,
    n_fs_unique_subjects: int,
    n_matched: int,
    n_unmatched: int,
) -> None:
    """Log subject counts at every filter stage for audit traceability."""
    logger.info("=" * 60)
    logger.info("CONSERVATION ACCOUNTING — 00_extract_merge.py")
    logger.info("  FreeSurfer sessions (raw):              %5d", n_fs_raw)
    logger.info("  FreeSurfer sessions (5.3 + QC pass):   %5d", n_fs_filtered)
    logger.info("  Unique subjects after FS filtering:     %5d", n_fs_unique_subjects)
    logger.info("  Subjects with clinical match:           %5d", n_matched)
    logger.info("  Subjects without clinical match:        %5d", n_unmatched)
    logger.info("  Check (should be 0): %d",
                n_fs_unique_subjects - n_matched - n_unmatched)
    logger.info("=" * 60)
def save_outputs(
    merged_raw: pd.DataFrame,
    longitudinal: pd.DataFrame,
    upstream_paths: list[Path],
) -> None:
    """Write both output files with sibling provenance metadata."""
    config.ensure_project_dirs()
 
    save_csv(merged_raw,   config.MERGED_RAW_FILE)
    save_csv(longitudinal, config.LONGITUDINAL_RAW_FILE)
 
    meta = build_run_metadata(
    script_name="00_extract_merge.py"
    )

    meta["project_version"] = config.PROJECT_VERSION
    meta["config_snapshot"] = {
        "MATCH_TOLERANCE_DAYS_PRIMARY": config.MATCH_TOLERANCE_DAYS_PRIMARY,
        "MATCH_TOLERANCE_DAYS_SENSITIVITY": config.MATCH_TOLERANCE_DAYS_SENSITIVITY,
        "FS_VERSION_FILTER": config.FS_VERSION_FILTER,
        "n_subjects_merged_raw": len(merged_raw),
        "n_rows_longitudinal": len(longitudinal),
    }
    meta["upstream_files"] = upstream_paths
    write_metadata(meta, config.MERGED_RAW_FILE)
 
    save_environment_snapshot(config.PACKAGE_VERSION_FILE)
    logger.info("Outputs written to %s", config.INTERIM_DIR)
def main() -> None:
    """Orchestrate the full extract-merge pipeline."""
    config.ensure_project_dirs()
    logger.info("Starting 00_extract_merge.py  (project version %s)", config.PROJECT_VERSION)
    logger.info("Primary matching tolerance: %d days", config.MATCH_TOLERANCE_DAYS_PRIMARY)
    fs_raw       = load_freesurfer()
    udsd1        = load_udsd1()
    udsb4        = load_udsb4()
    demographics = load_demographics()
    _centoloid = load_centiloid()
    _braak = load_braak()
    n_fs_raw = len(fs_raw)
    fs_with_days = parse_session_days(fs_raw)
    fs_filtered, _excluded_ids = apply_fs_filters(fs_with_days)
    n_fs_filtered = len(fs_filtered)
    n_fs_subjects = fs_filtered[config.ID_COL].nunique()

    fs_computed = compute_derived_biomarkers(fs_filtered)
    combined_clinical = build_combined_clinical(udsd1, udsb4)
    baseline = select_baseline_sessions(
        fs_computed,
        combined_clinical,
        tolerance_days=config.MATCH_TOLERANCE_DAYS_PRIMARY,
    )
 
    n_matched   = len(baseline)
    n_unmatched = n_fs_subjects - n_matched
    baseline_with_demo = merge_demographics(baseline, demographics)
    merged_raw = select_merged_raw_columns(baseline_with_demo)
    longitudinal = build_longitudinal_table(udsb4, baseline)
    validate_merged_raw(merged_raw)
    validate_longitudinal(longitudinal)
    log_conservation(
        n_fs_raw=n_fs_raw,
        n_fs_filtered=n_fs_filtered,
        n_fs_unique_subjects=n_fs_subjects,
        n_matched=n_matched,
        n_unmatched=n_unmatched,
    )
 
    upstream_paths = list(config.RAW_FILE_PATHS.values())
    save_outputs(merged_raw, longitudinal, upstream_paths)
 
    logger.info(
        "00_extract_merge.py complete.  "
        "merged_raw.csv: %d subjects | longitudinal_raw.csv: %d rows.",
        len(merged_raw), len(longitudinal),
    )
 
 
if __name__ == "__main__":
    main()


    