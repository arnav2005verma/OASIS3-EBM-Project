from __future__ import annotations 
import sys
from pathlib import Path
from typing import Any
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
 
import config
from utils.io_utils import (
    build_run_metadata,
    file_fingerprint,
    save_csv,
    save_json,
    save_pickle,
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
 
logger = get_logger(__name__, log_file=config.LOGS_DIR / "01_data_prep.log")

ANALYSIS_DATASET_PATH = config.PROCESSED_DIR / "analysis_dataset.csv"
QC_SUMMARY_PATH = config.PROCESSED_DIR / "qc_summary.csv"
METADATA_PATH = config.PROCESSED_DIR / "analysis_dataset_metadata.json"
RESID_SUMMARY_PATH = config.PROCESSED_DIR / "residualization_summary.csv"
RESID_MODELS_DIR = config.PROCESSED_DIR / "residualization_models"

_MRI_BIOMARKERS: list[str] = [
    "hippocampus_vol",
    "entorhinal_thickness",
    "fusiform_vol",
    "inferiortemporal_vol",
    "ventricular_vol",
    "whole_brain_vol",
]
 
_THICKNESS_BIOMARKERS: list[str] = [
    "entorhinal_thickness",
]
 
_VOLUME_BIOMARKERS: list[str] = [
    "hippocampus_vol",
    "fusiform_vol",
    "inferiortemporal_vol",
    "ventricular_vol",
    "whole_brain_vol",
]
 
_SKEWED_BIOMARKERS: list[str] = [
    "ventricular_vol",
]

_HIGHER_IS_ABNORMAL: list[str] = [
    "ventricular_vol",
    "Centiloid_fSUVR_TOT_CORTMEAN",
    "Tauopathy",
]
 
_PET_BIOMARKERS: list[str] = [
    "Centiloid_fSUVR_TOT_CORTMEAN",
    "Tauopathy",
]

_LOWER_IS_ABNORMAL = [
    "hippocampus_vol",
    "entorhinal_thickness",
    "fusiform_vol",
    "inferiortemporal_vol",
    "whole_brain_vol",
]
 
_REQUIRED_COLS: list[str] = [
    config.ID_COL,
    "mri_days",
    "age_at_baseline",
    "sex",
    "education_years",
    "icv",
    config.MMSE_COL,
    config.CDR_GLOBAL_COL,
    config.NORMCOG_COL,
    config.DEMENTED_COL,
    config.PROBAD_COL,
    config.POSSAD_COL,
    *_MRI_BIOMARKERS,
]

_ALL_DX_LABELS: list[str] = [
    config.DX_CN,
    config.DX_CIND,
    config.DX_AD,
    config.DX_OTHER,
    config.DX_UNCLASSIFIED,
]

def _sign_check(
    df: pd.DataFrame,
    biomarker: str,
    raw_col: str,
    z_col: str,
    higher_is_abnormal: bool,
) -> None:
    """Verify the direction of a z-scored biomarker against its raw values.
 
    Splits the dataframe into the bottom and top quartile by *raw_col* and
    confirms the expected relationship holds between raw values and the
    natural (pre-flip) z-score:
 
    * ``higher_is_abnormal=True``  → high raw  → high z  (ventricles)
    * ``higher_is_abnormal=False`` → low raw   → low z   (volumes, thickness)
 
    Called before direction flipping.  Raises ``ValidationError`` loudly so
    the pipeline cannot proceed silently with a corrupted sign convention.
 
    Args:
        df:                  DataFrame with *raw_col* and *z_col* present.
        biomarker:           Name used in error messages.
        raw_col:             Column of untransformed raw values.
        z_col:               Column of natural z-scores (before direction flip).
        higher_is_abnormal:  True for ventricular_vol; False for all others.
 
    Raises:
        ValidationError: If the directional check fails.
    """
    q25 = df[raw_col].quantile(0.25)
    q75 = df[raw_col].quantile(0.75)

    mean_z_low = df.loc[df[raw_col] <= q25, z_col].mean()
    mean_z_high = df.loc[df[raw_col] >= q75, z_col].mean()

    if higher_is_abnormal:
         ok = mean_z_high > mean_z_low
    else:
        ok = mean_z_low < mean_z_high

    if not ok:
        raise ValidationError(
            f"Sign-check FAILED for '{biomarker}': "
            f"higher_is_abnormal={higher_is_abnormal}, "
            f"z_low-quartile mean={mean_z_low:.3f}, "
            f"z_high-quartile mean={mean_z_high:.3f}. "
            "Do not proceed to EBM fitting until this is resolved."
        )
 
    logger.info(
        "Sign-check PASSED '%s' (higher_is_abnormal=%s): "
        "z_low=%.3f  z_high=%.3f",
        biomarker, higher_is_abnormal, mean_z_low, mean_z_high,
    )

def _source_col(biomarker: str) -> str:
    """Return the column fed into the OLS formula for *biomarker*.
 
    Skewed biomarkers are log1p-transformed before residualization;
    all others use the raw column directly.
    """
    return f"log1p_{biomarker}" if biomarker in _SKEWED_BIOMARKERS else biomarker

def _residualize_formula(biomarker: str) -> str:
    """Return the patsy OLS formula string for *biomarker*.
 
    Volumes include ICV; thickness measures do not (per the OASIS-3 imaging
    data dictionary: thickness does not vary significantly with head size).
    """
    src = _source_col(biomarker)
    if biomarker in _THICKNESS_BIOMARKERS:
        return config.THICKNESS_RESIDUALIZATION_FORMULA.format(biomarker=src)
    return config.VOLUME_RESIDUALIZATION_FORMULA.format(biomarker=src)

def load_inputs() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load merged_raw.csv and longitudinal_raw.csv.
 
    Returns:
        (merged, longitudinal) DataFrames.
 
    Raises:
        FileNotFoundError: If either file is missing.
        ValidationError:   If required columns are absent or IDs are duplicated.
    """
    merged_path = config.INTERIM_DIR / "merged_raw.csv"
    long_path   = config.INTERIM_DIR / "longitudinal_raw.csv"
 
    if not merged_path.exists():
        raise FileNotFoundError(
            f"merged_raw.csv not found at {merged_path}. "
            "Run 00_extract_merge.py first."
        )
 
    merged = pd.read_csv(merged_path, low_memory=False)
    log_dataframe_shape(logger, merged, "merged_raw")
 
    assert_columns_present(merged, _REQUIRED_COLS)
    assert_no_duplicate_id(merged, config.ID_COL)
    assert_non_empty_dataframe(merged, "merged_raw")
 
    if not long_path.exists():
        logger.warning(
            "longitudinal_raw.csv not found at %s. "
            "Follow-up flags will be set to 0 for all subjects.", long_path,
        )
        longitudinal = pd.DataFrame(columns=[config.ID_COL, "years_since_baseline"])
    else:
        longitudinal = pd.read_csv(long_path, low_memory=False)
        log_dataframe_shape(logger, longitudinal, "longitudinal_raw")
 
    return merged, longitudinal
 
def derive_diagnosis_groups(df: pd.DataFrame) -> pd.DataFrame:
    """Assign diagnosis_group from UDSd1 binary flag columns.
 
    Priority order (highest priority applied last, overwriting lower-priority
    assignments):
 
        6  remaining DEMENTED without AD or exclusion label → OTHER_Dementia
        4  non-demented with any MCI flag → CIND
        3  DEMENTED and (PROBAD or POSSAD) → AD
        2  NORMCOG == 1 and not DEMENTED → CN
        1  DEMENTED and any non-AD etiology flag → OTHER_Dementia
 
    NaN in flag columns is treated as 0 (absent).
 
    Args:
        df: DataFrame with NORMCOG, DEMENTED, PROBAD, POSSAD, MCI flags,
            and non-AD exclusion flag columns present.
 
    Returns:
        Copy of df with 'diagnosis_group' column added.
    """
    df = df.copy()
 
    flag_cols = (
        [config.NORMCOG_COL, config.DEMENTED_COL, config.PROBAD_COL, config.POSSAD_COL]
        + config.MCI_FLAG_COLS
        + config.NON_AD_EXCLUSION_COLS
    )
    available = [c for c in flag_cols if c in df.columns]
    df[available] = df[available].fillna(0)
 
    avail_mci  = [c for c in config.MCI_FLAG_COLS      if c in df.columns]
    avail_excl = [c for c in config.NON_AD_EXCLUSION_COLS if c in df.columns]
 
    if not avail_mci:
        raise ValidationError("No MCI flag columns found in merged_raw.csv.")
    if not avail_excl:
        raise ValidationError("No non-AD exclusion columns found in merged_raw.csv.")
 
    demented   = df[config.DEMENTED_COL].eq(1)
    normcog    = df[config.NORMCOG_COL].eq(1)
    probad     = df[config.PROBAD_COL].eq(1)
    possad     = df[config.POSSAD_COL].eq(1)
    any_non_ad = df[avail_excl].eq(1).any(axis=1)
 
    if config.IMPNOMCI_INCLUDED_IN_CIND:
        any_mci = df[avail_mci].eq(1).any(axis=1)
    else:
        classical = [c for c in avail_mci if c != "IMPNOMCI"]
        any_mci   = df[classical].eq(1).any(axis=1) if classical else pd.Series(
            False, index=df.index
        )
        logger.info(
            "IMPNOMCI excluded from CIND (config.IMPNOMCI_INCLUDED_IN_CIND=False). "
            "Classical MCI columns used: %s", classical,
        )
 
    groups = pd.Series(config.DX_UNCLASSIFIED, index=df.index, dtype=str)
    groups[demented & ~any_non_ad & ~(probad | possad)] = config.DX_OTHER 
    groups[~normcog & ~demented & any_mci] = config.DX_CIND 
    groups[demented & (probad | possad)]  = config.DX_AD   
    groups[normcog & ~demented] = config.DX_CN
    groups[demented & any_non_ad] = config.DX_OTHER 

    df["diagnosis_group"] = groups
    unexpected = set(df["diagnosis_group"].unique()) - set(_ALL_DX_LABELS)
    if unexpected:
        raise ValidationError(
            f"Unexpected diagnosis_group values: {unexpected}. "
            "All rows must map to one of {_ALL_DX_LABELS}."
        )
 
    for label in _ALL_DX_LABELS:
        n = (df["diagnosis_group"] == label).sum()
        logger.info("  diagnosis_group='%s': %d subjects", label, n)
 
    return df

def flag_adequate_followup(
    df: pd.DataFrame,
    longitudinal: pd.DataFrame,
) -> pd.DataFrame:
    """Add follow-up adequacy flags to the baseline dataframe.
 
    A post-baseline visit qualifies if years_since_baseline × 365.25 ≥
    config.MIN_FOLLOWUP_DAYS.  A subject has adequate follow-up if they have
    ≥ config.MIN_FOLLOWUP_VISITS qualifying visits.
 
    Adds
    ----
    n_followup_visits  int  — qualifying post-baseline visit count
    adequate_followup  int  — 1 if n_followup_visits ≥ MIN_FOLLOWUP_VISITS
 
    Args:
        df:           Baseline dataframe (one row per subject).
        longitudinal: Long-format post-baseline visit table.
 
    Returns:
        df copy with follow-up flag columns added.
    """
    df = df.copy()
 
    if "years_since_baseline" not in longitudinal.columns or longitudinal.empty:
        df["n_followup_visits"] = 0
        df["adequate_followup"] = 0
        logger.warning(
            "longitudinal_raw.csv missing or lacks 'years_since_baseline'; "
            "adequate_followup set to 0 for all subjects."
        )
        return df
 
    min_years = config.MIN_FOLLOWUP_DAYS / 365.25
    qualifying = longitudinal[longitudinal["years_since_baseline"] >= min_years]
    visit_counts = (
        qualifying.groupby(config.ID_COL)
        .size()
        .rename("n_followup_visits")
    )
 
    df["n_followup_visits"] = (
        df[config.ID_COL].map(visit_counts).fillna(0).astype(int)
    )
    df["adequate_followup"] = (
        df["n_followup_visits"] >= config.MIN_FOLLOWUP_VISITS
    ).astype(int)
 
    n_ok = int(df["adequate_followup"].sum())
    logger.info(
        "Follow-up flag: %d / %d subjects have ≥ %d qualifying visit(s) "
        "(≥ %d days post-baseline).",
        n_ok, len(df), config.MIN_FOLLOWUP_VISITS, config.MIN_FOLLOWUP_DAYS,
    )
    return df

def build_cn_reference_mask(df: pd.DataFrame) -> pd.Series:
    """Return a boolean mask for the strict CN reference group.
 
    Definition:
        NORMCOG == 1  AND  DEMENTED != 1  AND
        CDRTOT  == 0  AND  MMSE >= MMSE_CN_THRESHOLD
 
    The reference group is used exclusively for fitting residualization
    OLS models and computing z-score normalisation parameters.  It must
    NOT overlap with any diseased subjects to avoid pathological variance
    contaminating the normative distribution.
 
    Args:
        df: Baseline dataframe with diagnosis flags, CDRTOT, and MMSE.
 
    Returns:
        Boolean pd.Series, True for CN reference subjects.
 
    Raises:
        ValidationError: If the reference group is smaller than
                         config.MIN_CN_REFERENCE_N.
    """
    mask = (
        df[config.NORMCOG_COL].eq(1)
        & df[config.DEMENTED_COL].ne(1)
        & df[config.CDR_GLOBAL_COL].eq(config.CDR_CN_THRESHOLD)
        & df[config.MMSE_COL].ge(config.MMSE_CN_THRESHOLD)
    )
 
    n_cn = int(mask.sum())
    logger.info(
        "CN reference (%s==1, %s!=1, %s==%.0f, %s>=%.0f): %d subjects",
        config.NORMCOG_COL, config.DEMENTED_COL,
        config.CDR_GLOBAL_COL, config.CDR_CN_THRESHOLD,
        config.MMSE_COL, config.MMSE_CN_THRESHOLD,
        n_cn,
    )
 
    if n_cn < config.MIN_CN_REFERENCE_N:
        raise ValidationError(
            f"CN reference group too small: {n_cn} subjects "
            f"(minimum {config.MIN_CN_REFERENCE_N}). "
            "Check diagnosis flags and CN thresholds."
        )
 
    return mask
 
def prepare_covariates(df: pd.DataFrame) -> pd.DataFrame:
    """Add derived covariates required by the residualization formulas.
 
    Adds
    ----
    age_sq : float — age_at_baseline squared (quadratic age term)
 
    Warns if 'sex' or 'age_at_baseline' contain NaN; subjects with missing
    covariates will receive NaN residuals (and therefore NaN z-scores).
 
    Args:
        df: Baseline dataframe.
 
    Returns:
        df copy with 'age_sq' added.
    """
    df = df.copy()
    df["age_sq"] = df["age_at_baseline"] ** 2
 
    for col in ["age_at_baseline", "sex", "icv"]:
        n_null = int(df[col].isna().sum())
        if n_null:
            logger.warning(
                "'%s' has %d null value(s). Affected subjects will have "
                "NaN residuals and will not contribute to EBM fitting.", col, n_null,
            )
 
    return df
def apply_log_transforms(df: pd.DataFrame) -> pd.DataFrame:
    """Apply log1p transform to right-skewed biomarkers.
 
    Adds 'log1p_{biomarker}' for each biomarker in _SKEWED_BIOMARKERS.
    The original column is preserved.  Residualization uses the transformed
    column; all z-scores represent log-scale standardised residuals for
    skewed biomarkers.
 
    Args:
        df: Baseline dataframe.
 
    Returns:
        df copy with log1p columns added.
    """
    df = df.copy()
    for biomarker in _SKEWED_BIOMARKERS:
        if biomarker not in df.columns:
            logger.warning(
                "Skewed biomarker '%s' not in dataframe; skipping log transform.",
                biomarker,
            )
            continue
        log_col = f"log1p_{biomarker}"
        df[log_col] = np.log1p(df[biomarker])
        logger.info(
            "log1p('%s'): raw mean=%.0f sd=%.0f  → log mean=%.3f sd=%.3f",
            biomarker,
            df[biomarker].mean(), df[biomarker].std(),
            df[log_col].mean(), df[log_col].std(),
        )
    return df

def residualize_single(
    df: pd.DataFrame,
    biomarker: str,
    cn_mask: pd.Series,
) -> tuple[pd.Series, dict[str, Any], Any]:
    """Fit OLS on the CN reference and return residuals for all subjects.
 
    The model is fitted on CN-only data to prevent disease-related variance
    from contaminating the normative regression coefficients.  It is then
    applied to the full cohort.
 
    Args:
        df:        Baseline dataframe (all subjects).
        biomarker: Pipeline-internal biomarker name.
        cn_mask:   Boolean mask for the CN reference group.
 
    Returns:
        (residuals, coefficient_record, fitted_model)
 
        residuals         — pd.Series (NaN where covariates are missing).
        coefficient_record — dict with formula, params, R², n_cn_obs.
        fitted_model      — statsmodels RegressionResultsWrapper for pickling.
 
    Raises:
        ValidationError: If too few CN subjects have complete data.
    """
    src     = _source_col(biomarker)
    formula = _residualize_formula(biomarker)
 
    cn_df      = df[cn_mask].copy()
    covariate_cols = ["age_at_baseline", "age_sq", "sex", "icv", src]
    complete_cn    = cn_df.dropna(subset=covariate_cols)
    n_complete     = len(complete_cn)
 
    if n_complete < config.MIN_CN_REFERENCE_N:
        raise ValidationError(
            f"Only {n_complete} CN subjects have complete data for "
            f"'{biomarker}' residualization (minimum {config.MIN_CN_REFERENCE_N})."
        )
 
    model     = smf.ols(formula, data=complete_cn).fit()
    predicted = model.predict(df)
    residuals = df[src] - predicted
 
    coef_record: dict[str, Any] = {
        "biomarker": biomarker,
        "source_col": src,
        "formula": formula,
        "n_cn_obs": n_complete,
        "r_squared": float(model.rsquared),
        "adj_r_squared":float(model.rsquared_adj),
        "params": {k: float(v) for k, v in model.params.items()},
        "pvalues": {k: float(v) for k, v in model.pvalues.items()},
    }
 
    n_null = int(residuals.isna().sum())
    logger.info(
        "Residualized '%s': n_CN=%d, R²=%.3f%s",
        biomarker, n_complete, model.rsquared,
        f", {n_null} null residuals" if n_null else "",
    )
    return residuals, coef_record, model
 
 
def residualize_all_mri(
    df: pd.DataFrame,
    cn_mask: pd.Series,
) -> tuple[pd.DataFrame, list[dict[str, Any]], dict[str, Any]]:
    """Residualize all MRI biomarkers against the CN normative model.
 
    Adds '{biomarker}_resid' columns to the dataframe.
 
    Args:
        df:      Baseline dataframe (must contain log1p columns for skewed
                 biomarkers, added by apply_log_transforms).
        cn_mask: Boolean mask for the CN reference group.
 
    Returns:
        (df_with_residuals, coef_records, fitted_models)
 
        coef_records  — list of coefficient dicts (one per biomarker) for
                        the residualization summary CSV.
        fitted_models — {biomarker: model} for pickling.
    """
    df = df.copy()
    coef_records: list[dict[str, Any]] = []
    fitted_models: dict[str, Any] = {}
 
    for biomarker in _MRI_BIOMARKERS:
        residuals, coef_record, model = residualize_single(df, biomarker, cn_mask)
        df[f"{biomarker}_resid"] = residuals
        coef_records.append(coef_record)
        fitted_models[biomarker] = model
 
    return df, coef_records, fitted_models
 
def zscore_mri_biomarkers(
    df: pd.DataFrame,
    cn_mask: pd.Series,
    coef_records: list[dict[str, Any]],
) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    """Z-score MRI residuals and apply direction standardisation.
 
    For each biomarker:
      1. Compute natural z = (resid - CN_mean) / CN_sd
      2. Run sign_check on the natural z-score to confirm direction is
         consistent with the underlying raw values.
      3. Apply direction flip for LOWER_IS_ABNORMAL biomarkers so that
         *higher z always means more Alzheimer's pathology*.
 
    The column 'z_{biomarker}' in the output follows this convention for
    every MRI biomarker.
 
    Args:
        df:           Dataframe with '{biomarker}_resid' columns.
        cn_mask:      Boolean mask for the CN reference group.
        coef_records: List of coefficient records to augment with z-score
                      parameters in-place.
 
    Returns:
        (df_with_z_columns, augmented_coef_records)
 
    Raises:
        ValidationError: If CN SD is zero for any biomarker, or if any
                         sign_check fails.
    """
    df = df.copy()
    coef_by_name = {r["biomarker"]: r for r in coef_records}
 
    for biomarker in _MRI_BIOMARKERS:
        resid_col    = f"{biomarker}_resid"
        cn_residuals = df.loc[cn_mask, resid_col]
 
        cn_mean = float(cn_residuals.mean())
        cn_sd   = float(cn_residuals.std(ddof=1))
 
        if cn_sd == 0.0:
            raise ValidationError(
                f"CN residual SD is zero for '{biomarker}'. "
                "Residualization produced a constant series; cannot z-score."
            )
        z_natural = (df[resid_col] - cn_mean) / cn_sd
        tmp = f"__z_nat_{biomarker}"
        df[tmp] = z_natural
        raw_col = biomarker
        higher_is_abnormal = biomarker in _HIGHER_IS_ABNORMAL
        _sign_check(df, biomarker, raw_col, tmp, higher_is_abnormal)
        df[f"z_{biomarker}"] = z_natural if higher_is_abnormal else -z_natural
        df.drop(columns=[tmp], inplace=True)

        if biomarker in coef_by_name:
            coef_by_name[biomarker]["cn_resid_mean"]    = cn_mean
            coef_by_name[biomarker]["cn_resid_sd"]      = cn_sd
            coef_by_name[biomarker]["direction_flipped"] = not higher_is_abnormal
 
        logger.info(
            "Z-scored '%s': CN_mean=%.4f  CN_sd=%.4f  flipped=%s",
            biomarker, cn_mean, cn_sd, not higher_is_abnormal,
        )
 
    return df, list(coef_by_name.values())
 
 
def zscore_pet_biomarkers(
    df: pd.DataFrame,
    cn_mask: pd.Series,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Z-score PET biomarkers relative to the CN PET-available reference.
 
    Centiloid and Tauopathy are already normalised (Centiloid to a 0/100
    reference scale; Tauopathy as a SUVR ratio).  Only z-scoring relative
    to the CN distribution is applied — no residualization against
    demographics is performed.
 
    Both measures are HIGHER_IS_ABNORMAL: no direction flip is needed.
 
    Args:
        df:      Baseline dataframe with Centiloid and Tauopathy columns.
        cn_mask: Boolean mask for the strict CN reference group.
 
    Returns:
        (df_with_pet_z, pet_zscore_params)
    """
    df = df.copy()
    pet_params: dict[str, Any] = {}
 
    for col, z_col in [
        ("Centiloid_fSUVR_TOT_CORTMEAN", "z_Centiloid_fSUVR_TOT_CORTMEAN"),
        ("Tauopathy",                     "z_Tauopathy"),
    ]:
        if col not in df.columns:
            logger.info("PET column '%s' not in dataframe; z-score set to NaN.", col)
            df[z_col] = np.nan
            continue
 
        cn_pet = df.loc[cn_mask & df[col].notna(), col]
        n_cn_pet = len(cn_pet)
 
        if n_cn_pet < config.MIN_CN_REFERENCE_N:
            logger.warning(
                "Only %d CN subjects have '%s' data (minimum %d). "
                "Z-score will be computed but flagged as low-power.",
                n_cn_pet, col, config.MIN_CN_REFERENCE_N,
            )
 
        if n_cn_pet == 0:
            logger.warning(
                "No CN subjects have '%s' data. Setting '%s' to NaN.", col, z_col,
            )
            df[z_col] = np.nan
            continue
 
        cn_mean = float(cn_pet.mean())
        cn_sd   = float(cn_pet.std(ddof=1))
 
        if cn_sd == 0.0:
            raise ValidationError(f"CN SD for PET measure '{col}' is zero.")
        df[z_col] = (df[col] - cn_mean) / cn_sd
 
        pet_params[col] = {
            "n_cn_reference":   n_cn_pet,
            "cn_mean":          cn_mean,
            "cn_sd":            cn_sd,
            "direction_flipped": False,
        }
        logger.info(
            "PET z-score '%s': n_CN=%d  CN_mean=%.3f  CN_sd=%.3f",
            col, n_cn_pet, cn_mean, cn_sd,
        )
 
    return df, pet_params
def build_analysis_flags(
    df: pd.DataFrame,
    cn_mask: pd.Series,
) -> pd.DataFrame:
    """Add eligibility and panel-completeness flags used by downstream scripts.
 
    Adds
    ----
    cn_reference_flag            int8 — 1 if in strict CN reference
    ebm_eligible                 int8 — 1 if diagnosis_group ∈ EBM_INCLUDED_DX_GROUPS
    has_amyloid_pet              int8 — 1 if Centiloid_fSUVR_TOT_CORTMEAN non-null
    has_tau_pet                  int8 — 1 if Tauopathy non-null
    panel_mri_complete           int8 — complete data for MRI-only panel
    panel_mri_amyloid_complete   int8 — complete data for MRI + Amyloid panel
    panel_mri_tau_complete       int8 — complete data for MRI + Amyloid + Tau panel
 
    Args:
        df:      Analysis dataframe with z-scored columns.
        cn_mask: Boolean mask for the CN reference group.
 
    Returns:
        df copy with flag columns added.
    """
    df = df.copy()
 
    df["cn_reference_flag"] = cn_mask.astype(int)
    df["ebm_eligible"]      = df["diagnosis_group"].isin(
        config.EBM_INCLUDED_DX_GROUPS
    ).astype(int)
 
    df["has_amyloid_pet"] = df["Centiloid_fSUVR_TOT_CORTMEAN"].notna().astype(int)
    df["has_tau_pet"]     = df["Tauopathy"].notna().astype(int)
 
    mri_z_cols     = [f"z_{b}" for b in _MRI_BIOMARKERS]
    amyloid_z_col  = "z_Centiloid_fSUVR_TOT_CORTMEAN"
    tau_z_col      = "z_Tauopathy"
 
    df["panel_mri_complete"]           = df[mri_z_cols].notna().all(axis=1).astype(int)
    df["panel_mri_amyloid_complete"]   = df[mri_z_cols + [amyloid_z_col]].notna().all(axis=1).astype(int)
    df["panel_mri_tau_complete"]       = df[mri_z_cols + [amyloid_z_col, tau_z_col]].notna().all(axis=1).astype(int)
 
    ebm = df[df["ebm_eligible"].eq(1)]
    for flag in ["panel_mri_complete", "panel_mri_amyloid_complete", "panel_mri_tau_complete"]:
        n = int(ebm[flag].sum())
        logger.info("EBM-eligible subjects with '%s': %d", flag, n)
 
    return df

def build_qc_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Build a tabular QC summary of the analysis dataset.
 
    Returns:
        DataFrame with columns [category, subcategory, n, pct_of_total].
    """
    n_total = len(df)
 
    def row(cat: str, sub: str, n: int) -> dict[str, Any]:
        return {
            "category":     cat,
            "subcategory":  sub,
            "n":            n,
            "pct_of_total": round(n / n_total * 100, 1) if n_total else 0.0,
        }
 
    rows: list[dict[str, Any]] = [row("Total", "all_subjects", n_total)]
 
    for label in _ALL_DX_LABELS:
        rows.append(row("Diagnosis", label, int((df["diagnosis_group"] == label).sum())))
 
    rows.append(row("CN reference", "cn_reference", int(df["cn_reference_flag"].sum())))
    rows.append(row("EBM", "ebm_eligible", int(df["ebm_eligible"].sum())))
    rows.append(row("Follow-up", "adequate", int(df["adequate_followup"].sum())))
    rows.append(row("PET", "has_amyloid_pet", int(df["has_amyloid_pet"].sum())))
    rows.append(row("PET", "has_tau_pet", int(df["has_tau_pet"].sum())))
 
    for flag in ["panel_mri_complete", "panel_mri_amyloid_complete", "panel_mri_tau_complete"]:
        rows.append(row("Panel (all)",          flag, int(df[flag].sum())))
        n_ebm = int(df.loc[df["ebm_eligible"].eq(1), flag].sum())
        rows.append(row("Panel (EBM eligible)", flag, n_ebm))
    for biomarker in _MRI_BIOMARKERS:
        z_col  = f"z_{biomarker}"
        n_null = int(df[z_col].isna().sum()) if z_col in df.columns else n_total
        rows.append(row("Z-score null", z_col, n_null))
    if "Centiloid_fSUVR_TOT_CORTMEAN" in df.columns:
        n_pos = int((df["Centiloid_fSUVR_TOT_CORTMEAN"] > 20.6).sum())
        rows.append(row("Amyloid", "positive_AV45_cutoff", n_pos))
 
    return pd.DataFrame(rows)

def build_residualization_summary(coef_records: list[dict[str, Any]]) -> pd.DataFrame:
    """Build a summary CSV of OLS coefficients and fit statistics.
 
    Args:
        coef_records: List of coefficient records from residualize_all_mri.
 
    Returns:
        DataFrame suitable for saving to residualization_summary.csv.
    """
    rows: list[dict[str, Any]] = []
    for rec in coef_records:
        base = {
            "biomarker":        rec.get("biomarker"),
            "source_col":       rec.get("source_col"),
            "formula":          rec.get("formula"),
            "n_cn_obs":         rec.get("n_cn_obs"),
            "r_squared":        rec.get("r_squared"),
            "adj_r_squared":    rec.get("adj_r_squared"),
            "cn_resid_mean":    rec.get("cn_resid_mean", np.nan),
            "cn_resid_sd":      rec.get("cn_resid_sd",   np.nan),
            "direction_flipped": rec.get("direction_flipped", False),
        }
        for k, v in rec.get("params", {}).items():
            base[f"param_{k}"] = v
        rows.append(base)
    return pd.DataFrame(rows)

def log_exclusions(
    df: pd.DataFrame,
    exclusion_log: list[dict[str, Any]],
) -> None:
    """Append non-EBM-eligible subject counts to the in-memory exclusion log.
 
    Does NOT remove rows from df — the analysis dataset retains all subjects
    so they appear in Table 1 descriptives.
 
    Args:
        df:            Analysis dataset with 'diagnosis_group'.
        exclusion_log: In-memory list (modified in-place by append_exclusion_log).
    """
    n_total = len(df)
 
    n_ebm     = int(df["ebm_eligible"].sum())
    n_other   = int((df["diagnosis_group"] == config.DX_OTHER).sum())
    n_unclass = int((df["diagnosis_group"] == config.DX_UNCLASSIFIED).sum())
 
    if n_other:
        append_exclusion_log(
            exclusion_log,
            "dx_exclusion_non_ad_dementia",
            n_total,
            n_total - n_other,
        )
        logger.info(
            "EBM exclusion: %d subjects with non-AD dementia (diagnosis_group='%s').",
            n_other, config.DX_OTHER,
        )
 
    if n_unclass:
        append_exclusion_log(
            exclusion_log,
            "dx_exclusion_unclassified",
            n_total - n_other,
            n_ebm,
        )
        logger.info(
            "EBM exclusion: %d subjects with unclassified diagnosis.",
            n_unclass,
        )
 
    log_cascading_filter(
        logger, "ebm_eligible", n_total, n_ebm
    )
    n_non_cn_ebm = int(
        df[df["ebm_eligible"].eq(1) & df["diagnosis_group"].ne(config.DX_CN)].shape[0]
    )
    if n_non_cn_ebm < config.MIN_N_FOR_EBM:
        msg = (
            f"Only {n_non_cn_ebm} non-CN EBM-eligible subjects (CIND + AD) — "
            f"below recommended minimum of {config.MIN_N_FOR_EBM}. "
            "EBM results should be interpreted as exploratory."
        )
        log_warning_file(msg, config.COHORT_SIZE_WARNING_FILE)
        logger.warning(msg)
 
def validate_analysis_dataset(df: pd.DataFrame) -> None:
    """Run structural and scientific validation on the final dataset.
 
    Raises:
        ValidationError: On any structural failure.
    """
    assert_no_duplicate_id(df, config.ID_COL)
    assert_non_empty_dataframe(df, "analysis_dataset")
    assert_columns_present(
        df,
        ["diagnosis_group", "cn_reference_flag", "ebm_eligible",
         "adequate_followup"] + [f"z_{b}" for b in _MRI_BIOMARKERS],
    )
 
    # Every diagnosis label must be from the allowed set
    unexpected = set(df["diagnosis_group"].unique()) - set(_ALL_DX_LABELS)
    if unexpected:
        raise ValidationError(
            f"Unexpected diagnosis_group values after full pipeline: {unexpected}"
        )
    cn_mask = df["cn_reference_flag"].eq(1)
    for biomarker in _MRI_BIOMARKERS:
        z_col = f"z_{biomarker}"
        cn_z  = df.loc[cn_mask, z_col]
        if cn_z.notna().sum() < 5:
            continue
        cn_mean_z = float(cn_z.mean())
        cn_sd_z   = float(cn_z.std())
        if abs(cn_mean_z) > 0.05:
            logger.warning(
                "CN z-mean for '%s' is %.4f (expected ≈ 0). "
                "Possible residualization or sign error.", z_col, cn_mean_z,
            )
        if abs(cn_sd_z - 1.0) > 0.05:
            logger.warning(
                "CN z-sd for '%s' is %.4f (expected ≈ 1.0).", z_col, cn_sd_z,
            )
 
    logger.info(
        "validate_analysis_dataset: PASSED — %d subjects × %d columns.",
        len(df), df.shape[1],
    )

def save_outputs(
    df: pd.DataFrame,
    qc_summary: pd.DataFrame,
    resid_summary: pd.DataFrame,
    coef_records: list[dict[str, Any]],
    pet_zscore_params: dict[str, Any],
    fitted_models: dict[str, Any],
    exclusion_log: list[dict[str, Any]],
    upstream_paths: list[Path],
) -> None:
    """Write all pipeline outputs for this script.
 
    Files written
    -------------
    data/processed/analysis_dataset.csv
    data/processed/analysis_dataset_metadata.json
    data/processed/qc_summary.csv
    data/processed/residualization_summary.csv
    data/processed/residualization_models/{biomarker}.pkl
    """
    config.ensure_project_dirs()
    RESID_MODELS_DIR.mkdir(parents=True, exist_ok=True)

    save_csv(df, ANALYSIS_DATASET_PATH)
    save_csv(qc_summary,QC_SUMMARY_PATH)
    save_csv(resid_summary, RESID_SUMMARY_PATH)

    for biomarker, model in fitted_models.items():
        save_pickle(model, RESID_MODELS_DIR / f"{biomarker}.pkl")
    
    meta = build_run_metadata("01_data_prep.py")
    meta.update({
        "project_version": config.PROJECT_VERSION,
        "n_subjects": len(df),
        "n_ebm_eligible": int(df["ebm_eligible"].sum()),
        "n_cn_reference":int(df["cn_reference_flag"].sum()),
        "n_adequate_followup": int(df["adequate_followup"].sum()),
        "n_amyloid_pet": int(df["has_amyloid_pet"].sum()),
        "n_tau_pet": int(df["has_tau_pet"].sum()),
        "mri_biomarkers": _MRI_BIOMARKERS,
        "thickness_biomarkers": _THICKNESS_BIOMARKERS,
        "volume_biomarkers": _VOLUME_BIOMARKERS,
        "lower_is_abnormal": _LOWER_IS_ABNORMAL,
        "higher_is_abnormal": _HIGHER_IS_ABNORMAL,
        "cn_definition": {
            "normcog_eq": 1,
            "demented_ne": 1,
            "cdrtot_eq": config.CDR_CN_THRESHOLD,
            "mmse_ge": config.MMSE_CN_THRESHOLD,
        },
        "impnomci_included_in_cind": config.IMPNOMCI_INCLUDED_IN_CIND,
        "residualization_coefficients": {
            r["biomarker"]: {
                "formula": r.get("formula"),
                "n_cn_obs": r.get("n_cn_obs"),
                "r_squared": r.get("r_squared"),
                "direction_flipped": r.get("direction_flipped"),
            }
            for r in coef_records
        },
        "pet_zscore_params": pet_zscore_params,
        "exclusion_log": exclusion_log,
        "upstream_fingerprints": {
            str(p): file_fingerprint(p) if p.exists() else "MISSING"
            for p in upstream_paths
        },
    })
 
    write_metadata(meta, METADATA_PATH)
 
    logger.info(
        "Outputs saved:  %s  |  %s  |  %s  |  %s",
        ANALYSIS_DATASET_PATH.name,
        QC_SUMMARY_PATH.name,
        RESID_SUMMARY_PATH.name,
        METADATA_PATH.name,
    )
 
def main() -> None:
    """Orchestrate the full data preparation pipeline."""
    config.ensure_project_dirs()
 
    logger.info("=" * 70)
    logger.info(
        "01_data_prep.py  |  project version %s", config.PROJECT_VERSION
    )
    logger.info("=" * 70)
 
    exclusion_log: list[dict[str, Any]] = []
    merged, longitudinal = load_inputs()
    n_input = len(merged)
    logger.info("Input: %d subjects from merged_raw.csv", n_input)
    logger.info("--- Diagnosis derivation ---")
    df = derive_diagnosis_groups(merged)
    logger.info("--- Follow-up flags ---")
    df = flag_adequate_followup(df, longitudinal)
    logger.info("--- CN reference group ---")
    cn_mask = build_cn_reference_mask(df)
    df = prepare_covariates(df)
    logger.info("--- Log transforms ---")
    df = apply_log_transforms(df)
    logger.info("--- MRI residualization ---")
    df, coef_records, fitted_models = residualize_all_mri(df, cn_mask)
    logger.info("--- MRI z-scoring ---")
    df, coef_records = zscore_mri_biomarkers(df, cn_mask, coef_records)
    logger.info("--- PET z-scoring ---")
    df, pet_zscore_params = zscore_pet_biomarkers(df, cn_mask)
    logger.info("--- Analysis flags ---")
    df = build_analysis_flags(df, cn_mask)
    log_exclusions(df, exclusion_log)
    qc_summary = build_qc_summary(df)
    resid_summary = build_residualization_summary(coef_records)
 
    logger.info("QC Summary:\n%s", qc_summary.to_string(index=False))
    validate_analysis_dataset(df)
    upstream_paths = [
        config.INTERIM_DIR / "merged_raw.csv",
        config.INTERIM_DIR / "longitudinal_raw.csv",
    ]
 
    save_outputs(
        df=df,
        qc_summary=qc_summary,
        resid_summary=resid_summary,
        coef_records=coef_records,
        pet_zscore_params=pet_zscore_params,
        fitted_models=fitted_models,
        exclusion_log=exclusion_log,
        upstream_paths=upstream_paths,
    )
 
    logger.info(
        "01_data_prep.py complete — %d subjects × %d columns → %s",
        len(df), df.shape[1], ANALYSIS_DATASET_PATH,
    )
 
 
if __name__ == "__main__":
    main()
 

 
