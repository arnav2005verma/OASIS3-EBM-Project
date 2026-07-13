from __future__ import annotations
import sys
import warnings
from pathlib import Path
from typing import Any
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.formula.api as smf
import statsmodels.api as sm

warnings.filterwarnings("ignore")

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
    get_logger,
    log_dataframe_shape,
    log_cascading_filter,
    append_exclusion_log,
)
from utils.validation_utils import (
    ValidationError,
    assert_columns_present,
    assert_no_duplicate_id,
    assert_non_empty_dataframe,
)
 
logger = get_logger(__name__, log_file=config.LOGS_DIR / "04_longitudinal_validation.log")

PANEL_MRI     = "mri_only"
PANEL_AMYLOID = "mri_amyloid"
 
PANEL_STAGE_COLS = {
    PANEL_MRI:     "ebm_stage_mri",
    PANEL_AMYLOID: "ebm_stage_mri_amyloid",
}
 
PANEL_LABELS = {
    PANEL_MRI:     "MRI-only",
    PANEL_AMYLOID: "MRI+Amyloid",
}
OUTCOMES = {
    "MMSE":  config.MMSE_COL,
    "CDRSUM": config.CDR_SB_COL,
}

_PALETTE = {
    config.DX_CN:   "#4393C3",
    config.DX_CIND: "#F4A582",
    config.DX_AD:   "#D6604D",
}
_STAGE_CMAP = "YlOrRd"
_DPI = config.FIGURE_DPI

_REQUIRED_BASELINE = [
    config.ID_COL,
    "diagnosis_group",
    "adequate_followup",
    "n_followup_visits",
    "age_at_baseline",
    "sex",
    "education_years",
    "apoe4_carrier",
    config.MMSE_COL,
    config.CDR_GLOBAL_COL,
    config.CDR_SB_COL,
    "followup_years",
    "ebm_stage_mri",
]

_REQUIRED_LONG = [
    config.ID_COL,
    "years_since_baseline",
    config.MMSE_COL,
    config.CDR_SB_COL,
]

def load_baseline_dataset() -> pd.DataFrame:
    """Load the longitudinal validation dataset produced by 03_ebm_staging.py.
 
    Returns
    -------
    pd.DataFrame  One row per subject.
 
    Raises
    ------
    FileNotFoundError  If the file is missing.
    ValidationError    If required columns are absent or IDs are duplicated.
    """
    path = config.PROCESSED_DIR / "longitudinal_validation_dataset.csv"
    if not path.exists():
        raise FileNotFoundError(
            f"longitudinal_validation_dataset.csv not found: {path}\n"
            "Run 03_ebm_staging.py first."
        )
 
    df = pd.read_csv(path, low_memory=False)
    df = df.loc[:, ~df.columns.str.startswith("Unnamed")]
 
    log_dataframe_shape(logger, df, "longitudinal_validation_dataset")
 
    assert_columns_present(df, _REQUIRED_BASELINE)
    assert_no_duplicate_id(df, config.ID_COL)
    assert_non_empty_dataframe(df, "longitudinal_validation_dataset")
 
    n_with_stage = df["ebm_stage_mri"].notna().sum()
    logger.info(
        "Subjects with ebm_stage_mri=%d  ebm_stage_mri_amyloid=%d",
        n_with_stage,
        df["ebm_stage_mri_amyloid"].notna().sum() if "ebm_stage_mri_amyloid" in df else 0,
    )
    return df

def load_longitudinal_visits() -> pd.DataFrame | None:
    """Load per-visit longitudinal data from data/interim/longitudinal_raw.csv.
 
    Returns None if the file is absent (cross-sectional analyses remain valid;
    the mixed-effects models are skipped with a clear warning).
    """
    path = config.LONGITUDINAL_RAW_FILE
    if not path.exists():
        logger.warning(
            "longitudinal_raw.csv not found at %s. "
            "Mixed-effects models will be skipped; "
            "cross-sectional and conversion analyses will proceed.", path,
        )
        return None
 
    df = pd.read_csv(path, low_memory=False)
    log_dataframe_shape(logger, df, "longitudinal_raw")
    assert_columns_present(df, _REQUIRED_LONG)
    return df

def prepare_analysis_cohort(
    baseline: pd.DataFrame,
    panel_name: str,
    exclusion_log: list[dict[str, Any]],
) -> pd.DataFrame:
    """Return the analysis-ready cohort for one panel.
 
    Filters to:
      - adequate_followup == 1
      - non-null EBM stage for this panel
      - EBM-eligible diagnosis groups (CN, CIND, AD)
 
    Adds:
      - stage_tertile  (Low / Mid / High based on panel-specific max stage)
      - apoe4_str      (categorical string for table formatting)
 
    Args
    ----
    baseline     Full longitudinal validation dataset.
    panel_name   "mri_only" or "mri_amyloid".
    exclusion_log In-memory exclusion accounting list.
 
    Returns
    -------
    pd.DataFrame  Analysis cohort for this panel.
    """
    stage_col = PANEL_STAGE_COLS[panel_name]
    n_start   = len(baseline)
 
    df = baseline[baseline[config.ID_COL].notna()].copy()
    df = df[df["adequate_followup"] == 1].copy()
    log_cascading_filter(logger, f"{panel_name}_adequate_followup", n_start, len(df))
    append_exclusion_log(exclusion_log, f"{panel_name}_adequate_followup", n_start, len(df))

    n_before = len(df)
    df = df[df[stage_col].notna()].copy()
    log_cascading_filter(logger, f"{panel_name}_stage_not_null", n_before, len(df))
    append_exclusion_log(exclusion_log, f"{panel_name}_stage_not_null", n_before, len(df))
    n_before = len(df)
    df = df[df["diagnosis_group"].isin(config.EBM_INCLUDED_DX_GROUPS)].copy()
    log_cascading_filter(logger, f"{panel_name}_dx_filter", n_before, len(df))
    append_exclusion_log(exclusion_log, f"{panel_name}_dx_filter", n_before, len(df))
 
    if len(df) < config.MIN_N_FOR_LONGITUDINAL:
        raise ValidationError(
            f"Panel '{panel_name}': only {len(df)} subjects after filtering — "
            f"minimum required: {config.MIN_N_FOR_LONGITUDINAL}."
        )
    n_stages = int(df[stage_col].max())
    boundaries = [-1, n_stages // 3, 2 * n_stages // 3, n_stages]
    labels     = ["Low", "Mid", "High"]
    df["stage_tertile"] = pd.cut(
        df[stage_col], bins=boundaries, labels=labels
    )
    df["apoe4_str"] = df["apoe4_carrier"].map({0.0: "Non-ε4", 1.0: "ε4 carrier", np.nan: "Unknown"}).fillna("Unknown")
 
    logger.info(
        "[%s] Analysis cohort: N=%d  (CN=%d  CIND=%d  AD=%d)",
        panel_name, len(df),
        (df["diagnosis_group"] == config.DX_CN).sum(),
        (df["diagnosis_group"] == config.DX_CIND).sum(),
        (df["diagnosis_group"] == config.DX_AD).sum(),
    )
    return df
 
 
def build_longitudinal_long(
    visits: pd.DataFrame,
    cohort: pd.DataFrame,
    panel_name: str,
)-> pd.DataFrame:
    """Merge per-visit data with baseline stage and covariates.
 
    Only post-baseline visits (years_since_baseline > 0) are retained.
 
    Returns
    -------
    Long-format DataFrame: one row per (subject, follow-up visit).
    """
    stage_col = PANEL_STAGE_COLS[panel_name]
 
    baseline_cols = [
        config.ID_COL, stage_col,
        "diagnosis_group", "age_at_baseline", "sex",
        "education_years", "apoe4_carrier",
    ]
    available = [c for c in baseline_cols if c in cohort.columns]
    baseline_sub = cohort[available].copy()
    baseline_sub = baseline_sub.rename(columns={stage_col: "stage"})
 
    long = visits.merge(baseline_sub, on=config.ID_COL, how="inner")
    long = long[long["years_since_baseline"] > 0].copy()
    long = long.dropna(subset=["stage", "years_since_baseline"]).copy()
 
    log_dataframe_shape(logger, long, f"{panel_name}_long_format")
    return long

def compute_descriptive_stats(cohort: pd.DataFrame, panel_name: str) -> pd.DataFrame:
    """Compute cohort characteristics stratified by diagnosis group.
 
    Returns a wide-format summary table suitable for Table 1.
    """
    stage_col = PANEL_STAGE_COLS[panel_name]
 
    rows: list[dict[str, Any]] = []
 
    def _stat(series: pd.Series, fmt: str = "mean_sd") -> str:
        s = series.dropna()
        if len(s) == 0:
            return "—"
        if fmt == "mean_sd":
            return f"{s.mean():.2f} ± {s.std():.2f}"
        if fmt == "median_iqr":
            q25, q75 = s.quantile([0.25, 0.75])
            return f"{s.median():.1f} [{q25:.1f}–{q75:.1f}]"
        if fmt == "n_pct":
            n = int(s.sum())
            pct = 100.0 * n / len(series)
            return f"{n} ({pct:.1f}%)"
        return str(s.mean())
 
    groups = [
        ("All", cohort),
        ("CN", cohort[cohort["diagnosis_group"] == config.DX_CN]),
        ("CIND", cohort[cohort["diagnosis_group"] == config.DX_CIND]),
        ("AD", cohort[cohort["diagnosis_group"] == config.DX_AD]),
    ]
 
    variables = [
        ("N", lambda d: str(len(d)), "n"),
        ("Age (years)", lambda d: _stat(d["age_at_baseline"]), "mean_sd"),
        ("Female, n (%)", lambda d: _stat((d["sex"] == "F").astype(int), "n_pct"), "n_pct"),
        ("Education (yrs)", lambda d: _stat(d["education_years"]), "mean_sd"),
        ("APOE ε4, n (%)", lambda d: _stat(d["apoe4_carrier"], "n_pct"), "n_pct"),
        ("Baseline MMSE", lambda d: _stat(d[config.MMSE_COL]), "mean_sd"),
        ("Baseline CDR-SB", lambda d: _stat(d[config.CDR_SB_COL]), "mean_sd"),
        (f"EBM stage", lambda d: _stat(d[stage_col], "median_iqr"), "median_iqr"),
        ("Follow-up (yrs)", lambda d: _stat(d["followup_years"], "median_iqr"), "median_iqr"),
    ]
 
    for var_name, func, _ in variables:
        row = {"variable": var_name, "panel": panel_name}
        for gname, gdf in groups:
            try:
                row[gname] = func(gdf)
            except Exception:
                row[gname] = "—"
        rows.append(row)
 
    return pd.DataFrame(rows)

def cross_sectional_correlations(
    cohort: pd.DataFrame,
    panel_name: str,
) -> pd.DataFrame:
    """Spearman correlations between EBM stage and baseline cognitive measures.
 
    Analyses performed:
      - Stage vs baseline MMSE
      - Stage vs baseline CDR-SB
      - Stage vs baseline CDR global
 
    Additionally computes Kruskal-Wallis across diagnosis groups.
    """
    stage_col = PANEL_STAGE_COLS[panel_name]
    rows: list[dict[str, Any]] = []
 
    for outcome_label, outcome_col in [
        ("MMSE", config.MMSE_COL),
        ("CDR-SB", config.CDR_SB_COL),
        ("CDR global", config.CDR_GLOBAL_COL),
    ]:
        valid = cohort[[stage_col, outcome_col]].dropna()
        rho, p = stats.spearmanr(valid[stage_col], valid[outcome_col])
 
        n   = len(valid)
        r2  = rho ** 2
        f2  = r2 / (1 - r2)         
        r_ci_low, r_ci_high = _spearman_ci(rho, n)
 
        rows.append({
            "panel": panel_name,
            "analysis": "cross_sectional",
            "outcome": outcome_label,
            "n": n,
            "spearman_rho": round(rho, 4),
            "rho_ci_95_low": round(r_ci_low, 4),
            "rho_ci_95_high": round(r_ci_high, 4),
            "p_value": _format_p(p),
            "effect_f2": round(f2, 4),
        })
 
        logger.info(
            "[%s] Stage vs %s: rho=%.3f [%.3f, %.3f]  p=%s  n=%d",
            panel_name, outcome_label, rho, r_ci_low, r_ci_high, _format_p(p), n,
        )

    groups = [
        cohort.loc[cohort["diagnosis_group"] == dx, stage_col].dropna().values
        for dx in [config.DX_CN, config.DX_CIND, config.DX_AD]
        if (cohort["diagnosis_group"] == dx).any()
    ]
    if len(groups) >= 2:
        h, p_kw = stats.kruskal(*groups)
        logger.info(
            "[%s] Kruskal-Wallis stage ~ diagnosis: H=%.3f  p=%s",
            panel_name, h, _format_p(p_kw),
        )
        rows.append({
            "panel": panel_name,
            "analysis": "kruskal_wallis",
            "outcome": "diagnosis_group",
            "n": sum(len(g) for g in groups),
            "H_stat": round(h, 4),
            "p_value": _format_p(p_kw),
        })
 
    return pd.DataFrame(rows)

def fit_mixed_effects_model(
    long_df: pd.DataFrame,
    outcome_col: str,
    stage_col: str,
    panel_name: str,
    outcome_label: str,
) -> dict[str, Any] | None:
    """Fit a linear mixed-effects model predicting longitudinal cognitive change.
 
    Model:
        outcome ~ stage x years + age + C(sex) + education + (1 + years | subject)
 
    The stage x years interaction is the primary effect of interest: a
    significant positive (for CDR-SB) or negative (for MMSE) coefficient
    indicates that higher baseline EBM stage predicts faster decline.
 
    If the full random-slope model fails to converge, a random-intercept-only
    fallback is fitted and clearly flagged in the results.
 
    Args
    ----
    long_df       Long-format per-visit data with 'stage' and covariates.
    outcome_col   Column name of the outcome variable.
    stage_col     Original panel stage column name (for reference only).
    panel_name    Panel identifier string.
    outcome_label Human-readable outcome name.
 
    Returns
    -------
    dict of model results, or None if fitting completely fails.
    """
    required = [
        config.ID_COL, "stage", "years_since_baseline",
        outcome_col, "age_at_baseline", "sex", "education_years",
    ]
    missing = [c for c in required if c not in long_df.columns]
    if missing:
        logger.warning(
            "[%s] Cannot fit mixed model for %s — missing columns: %s",
            panel_name, outcome_label, missing,
        )
        return None
 
    analysis_df = long_df[required].dropna().copy()
    analysis_df["stage"]  = analysis_df["stage"].astype(float)
 
    n_subjects = analysis_df[config.ID_COL].nunique()
    n_obs = len(analysis_df)
 
    if n_subjects < config.MIN_N_FOR_LONGITUDINAL:
        logger.warning(
            "[%s] Too few subjects (%d) for mixed model (%s). Skipping.",
            panel_name, n_subjects, outcome_label,
        )
        return None
 
    formula = config.MIXED_MODEL_FORMULA_TEMPLATE.format(outcome=outcome_col)
    logger.info(
        "[%s] Fitting LME: %s  (n_subj=%d  n_obs=%d)",
        panel_name, formula, n_subjects, n_obs,
    )
 
    model_type = "full"
    result = None

    try:
        model = smf.mixedlm(
            formula,
            data=analysis_df,
            groups=analysis_df[config.ID_COL],
            re_formula=config.MIXED_MODEL_FULL_RE_FORMULA,
        )
        fit = model.fit(method="lbfgs", maxiter=2000)
        if not fit.converged:
            raise RuntimeError("Convergence flag False")
        result = fit
        logger.info("[%s] Full random-slope model converged.", panel_name)
    except Exception as exc:
        logger.warning(
            "[%s] Full random-slope model failed (%s). Trying fallback.", panel_name, exc,
        )
    
    if result is None:
        try:
            model = smf.mixedlm(
                formula,
                data=analysis_df,
                groups=analysis_df[config.ID_COL],
                re_formula=config.MIXED_MODEL_FALLBACK_RE_FORMULA,
            )
            fit   = model.fit(method="lbfgs", maxiter=2000)
            result = fit
            model_type = "fallback_random_intercept"
            logger.warning(
                "[%s] Using random-intercept fallback for %s.", panel_name, outcome_label,
            )
        except Exception as exc2:
            logger.error(
                "[%s] Mixed-effects model completely failed for %s: %s",
                panel_name, outcome_label, exc2,
            )
            return None
 
    return _extract_model_results(
        result, panel_name, outcome_label, outcome_col,
        model_type, n_subjects, n_obs, formula,
    )
 
 
def _extract_model_results(
    fit:           Any,
    panel_name:    str,
    outcome_label: str,
    outcome_col:   str,
    model_type:    str,
    n_subjects:    int,
    n_obs:         int,
    formula:       str,
) -> dict[str, Any]:
    """Extract coefficients, CIs, and effect sizes from a fitted MixedLM result."""
    ci    = fit.conf_int()
    rows: list[dict[str, Any]] = []
 
    for param in fit.params.index:
        beta = float(fit.params[param])
        se = float(fit.bse[param])
        t = float(fit.tvalues[param])
        p = float(fit.pvalues[param])
        lo = float(ci.loc[param, 0])
        hi = float(ci.loc[param, 1])

        rows.append({
            "panel": panel_name,
            "outcome": outcome_label,
            "model_type": model_type,
            "parameter": param,
            "beta": round(beta, 6),
            "SE": round(se, 6),
            "t_value": round(t, 4),
            "ci_95_low": round(lo, 6),
            "ci_95_high": round(hi, 6),
            "p_value": _format_p(p),
            "significant": p < 0.05,
            "n_subjects": n_subjects,
            "n_observations": n_obs,
            "formula": formula,
        })
 
        if "stage" in param.lower():
            logger.info(
                "[%s] %s  %s: β=%.4f (SE=%.4f) 95%%CI=[%.4f,%.4f]  p=%s",
                panel_name, outcome_label, param, beta, se, lo, hi, _format_p(p),
            )
 
    return {
        "rows": rows,
        "fit": fit,
        "n_subjects": n_subjects,
        "n_obs": n_obs,
        "aic": float(fit.aic) if hasattr(fit, "aic") else None,
    }
 
 
def compute_predicted_trajectories(
    long_df: pd.DataFrame,
    fit_result: dict[str, Any],
    outcome_col: str,
    panel_name: str,
    outcome_label: str,
    n_stages: int = 5,
) -> pd.DataFrame:
    """Generate model-predicted trajectories over time, stratified by stage.
 
    Produces predictions at a grid of time-points for stage values 0, n/2,
    and n (Low, Mid, High) using the fitted LME coefficients.
 
    Returns
    -------
    DataFrame with columns: panel, outcome, stage_level, years, predicted.
    """
    if fit_result is None:
        return pd.DataFrame()
 
    fit  = fit_result["fit"]
    time_grid = np.linspace(0.5, 8, 30)
    ref_age = float(long_df["age_at_baseline"].mean())
    ref_edu = float(long_df["education_years"].mean())
    ref_sex = "F"     # reference category from C(sex)
 
    rows: list[dict[str, Any]] = []
    for stage_val in [0, n_stages // 2, n_stages]:
        for t in time_grid:
            pred_df = pd.DataFrame({
                "years_since_baseline": [t],
                "stage":                [float(stage_val)],
                "age_at_baseline":      [ref_age],
                "sex":                  [ref_sex],
                "education_years":      [ref_edu],
            })
            try:
                predicted = float(fit.predict(pred_df).iloc[0])
            except Exception:
                predicted = float("nan")
            rows.append({
                "panel":        panel_name,
                "outcome":      outcome_label,
                "stage_level":  f"Stage {stage_val}",
                "years":        round(float(t), 3),
                "predicted":    round(predicted, 4),
            })
 
    return pd.DataFrame(rows)

def conversion_analysis(
    cohort: pd.DataFrame,
    panel_name: str,
) -> pd.DataFrame:
    """Logistic regression: does baseline EBM stage predict AD/dementia status?
 
    Since per-visit follow-up diagnosis data are not available in the
    baseline-only file, this analysis tests:
      - Stage predicting current AD diagnosis (binary: AD vs CN)
      - Stage predicting any impairment (binary: CIND+AD vs CN)
 
    This is a clinically meaningful proxy: it demonstrates that the EBM
    stage captures a disease state that is strongly linked to clinical
    diagnosis, beyond what would be expected by chance.
 
    A note on interpretation: because EBM stages are derived from cross-
    sectional MRI data and the diagnosis is available at the same baseline
    visit, this is a concurrent validity analysis, not a prospective
    prediction.  True prospective conversion analysis requires per-visit
    follow-up diagnosis data (available in longitudinal_raw if clinical
    diagnoses were recorded at each follow-up visit).
 
    Returns
    -------
    DataFrame of logistic regression results.
    """
    stage_col = PANEL_STAGE_COLS[panel_name]
    rows: list[dict[str, Any]] = []
 
    for outcome_name, outcome_mask, n_col in [
        ("AD_vs_CN",
         cohort["diagnosis_group"].isin([config.DX_AD, config.DX_CN]),
         "n_ad_cn"),
        ("Impaired_vs_CN",
         cohort["diagnosis_group"].isin([config.DX_CN, config.DX_CIND, config.DX_AD]),
         "n_all"),
    ]:
        sub = cohort.loc[outcome_mask].dropna(
            subset=[stage_col, "age_at_baseline", "sex", "education_years"]
        ).copy()
 
        if outcome_name == "AD_vs_CN":
            sub["outcome_bin"] = (sub["diagnosis_group"] == config.DX_AD).astype(int)
        else:
            sub["outcome_bin"] = (sub["diagnosis_group"] != config.DX_CN).astype(int)
 
        n_pos = int(sub["outcome_bin"].sum())
        n_neg = len(sub) - n_pos
        if n_pos < 10 or n_neg < 10:
            logger.warning(
                "[%s] Conversion (%s): too few events (n_pos=%d). Skipping.",
                panel_name, outcome_name, n_pos,
            )
            continue
 
        sub = sub.rename(columns={stage_col: "stage"})
        formula = "outcome_bin ~ stage + age_at_baseline + C(sex) + education_years"
 
        try:
            logit = smf.logit(formula, data=sub).fit(disp=False, maxiter=200)
        except Exception as exc:
            logger.warning(
                "[%s] Logistic regression failed (%s): %s", panel_name, outcome_name, exc
            )
            continue
 
        ci = logit.conf_int()
        for param in logit.params.index:
            beta = float(logit.params[param])
            p = float(logit.pvalues[param])
            lo = float(ci.loc[param, 0])
            hi = float(ci.loc[param, 1])
            or_val = float(np.exp(beta))
            or_ci_lo = float(np.exp(lo))
            or_ci_hi = float(np.exp(hi))
 
            rows.append({
                "panel": panel_name,
                "outcome": outcome_name,
                "parameter": param,
                "beta": round(beta, 6),
                "OR": round(or_val, 4),
                "OR_ci_95_low": round(or_ci_lo, 4),
                "OR_ci_95_high": round(or_ci_hi, 4),
                "p_value": _format_p(p),
                "significant": p < 0.05,
                "n_total": len(sub),
                "n_positive": n_pos,
            })
 
            if "stage" in param.lower():
                logger.info(
                    "[%s] Logistic (%s)  %s: OR=%.3f [%.3f–%.3f]  p=%s",
                    panel_name, outcome_name, param,
                    or_val, or_ci_lo, or_ci_hi, _format_p(p),
                )
        null_ll  = logit.llnull
        full_ll  = logit.llf
        pseudo_r2 = float(1 - (full_ll / null_ll))
        logger.info(
            "[%s] Logistic (%s): pseudo-R²=%.4f  n=%d",
            panel_name, outcome_name, pseudo_r2, len(sub),
        )
 
    return pd.DataFrame(rows)

def sensitivity_analyses(
    cohort: pd.DataFrame,
    visits: pd.DataFrame | None,
    panel_name: str,
) -> pd.DataFrame:
    """Run pre-specified sensitivity checks.
 
    1. CN-only subgroup: stage vs follow-up MMSE change
       (tests whether the effect holds even within the cognitively normal group)
    2. Without APOE adjustment: re-run primary Spearman without APOE covariate
    3. Older subjects only (age > 70): checks robustness to age confounding
 
    Returns
    -------
    DataFrame of sensitivity results (Spearman rho, p) for each check.
    """
    stage_col = PANEL_STAGE_COLS[panel_name]
    rows: list[dict[str, Any]] = []

    cn_sub = cohort[cohort["diagnosis_group"] == config.DX_CN].copy()
    for outcome_label, outcome_col in [("MMSE", config.MMSE_COL),
                                        ("CDR-SB", config.CDR_SB_COL)]:
        valid = cn_sub[[stage_col, outcome_col]].dropna()
        if len(valid) > 20:
            rho, p = stats.spearmanr(valid[stage_col], valid[outcome_col])
            rows.append({
                "panel": panel_name, "sensitivity_check": "CN_only",
                "outcome": outcome_label, "n": len(valid),
                "spearman_rho": round(rho, 4), "p_value": _format_p(p),
            })
            logger.info(
                "[%s] Sensitivity CN-only vs %s: rho=%.3f  p=%s  n=%d",
                panel_name, outcome_label, rho, _format_p(p), len(valid),
            )
    
    old_sub = cohort[cohort["age_at_baseline"] > 70].copy()
    for outcome_label, outcome_col in [("MMSE", config.MMSE_COL),
                                        ("CDR-SB", config.CDR_SB_COL)]:
        valid = old_sub[[stage_col, outcome_col]].dropna()
        if len(valid) > 20:
            rho, p = stats.spearmanr(valid[stage_col], valid[outcome_col])
            rows.append({
                "panel": panel_name, "sensitivity_check": "age_over_70",
                "outcome": outcome_label, "n": len(valid),
                "spearman_rho": round(rho, 4), "p_value": _format_p(p),
            })
            logger.info(
                "[%s] Sensitivity age>70 vs %s: rho=%.3f  p=%s  n=%d",
                panel_name, outcome_label, rho, _format_p(p), len(valid),
            )
    
    for apoe_val, apoe_label in [(0.0, "APOE_e4_neg"), (1.0, "APOE_e4_pos")]:
        apoe_sub = cohort[cohort["apoe4_carrier"] == apoe_val].copy()
        valid_mmse = apoe_sub[[stage_col, config.MMSE_COL]].dropna()
        if len(valid_mmse) > 20:
            rho, p = stats.spearmanr(valid_mmse[stage_col], valid_mmse[config.MMSE_COL])
            rows.append({
                "panel": panel_name, "sensitivity_check": apoe_label,
                "outcome": "MMSE", "n": len(valid_mmse),
                "spearman_rho": round(rho, 4), "p_value": _format_p(p),
            })
            logger.info(
                "[%s] Sensitivity %s vs MMSE: rho=%.3f  p=%s  n=%d",
                panel_name, apoe_label, rho, _format_p(p), len(valid_mmse),
            )
 
    return pd.DataFrame(rows)

def _fig_path(name: str) -> Path:
    config.FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    return config.FIGURES_DIR / name
 
 
def _apply_pub_style(ax: plt.Axes) -> None:
    """Apply consistent publication-quality formatting to an axes object."""
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(labelsize=10)
    ax.set_xlabel(ax.get_xlabel(), fontsize=11)
    ax.set_ylabel(ax.get_ylabel(), fontsize=11)
    if ax.get_title():
        ax.set_title(ax.get_title(), fontsize=12, fontweight="bold", pad=8)
 
 
def fig_stage_histogram(cohort: pd.DataFrame, panel_name: str) -> Path:
    """Histogram of EBM stage distribution, coloured by diagnosis group."""
    stage_col = PANEL_STAGE_COLS[panel_name]
    label     = PANEL_LABELS[panel_name]
 
    fig, ax = plt.subplots(figsize=(7, 4))
    stage_vals = sorted(cohort[stage_col].dropna().unique())
    dx_order = [config.DX_CN, config.DX_CIND, config.DX_AD]
    bottoms  = np.zeros(len(stage_vals))
    for dx in dx_order:
        sub = cohort[cohort["diagnosis_group"] == dx]
        counts = [int((sub[stage_col] == s).sum()) for s in stage_vals]
        ax.bar(
            stage_vals, counts, bottom=bottoms,
            color=_PALETTE.get(dx, "grey"), alpha=0.85,
            label=dx, edgecolor="white", linewidth=0.5,
        )
        bottoms += np.array(counts, dtype=float)
 
    ax.set_xlabel("EBM Stage", fontsize=11)
    ax.set_ylabel("Number of Subjects", fontsize=11)
    ax.set_title(f"EBM Stage Distribution — {label}", fontsize=12, fontweight="bold")
    ax.set_xticks(stage_vals)
    ax.legend(title="Diagnosis", fontsize=9, title_fontsize=9)
    _apply_pub_style(ax)
    plt.tight_layout()
 
    out = _fig_path(f"fig_stage_histogram_{panel_name}.png")
    fig.savefig(out, dpi=_DPI, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved: %s", out)
    return out
 
 
def fig_stage_by_diagnosis(cohort: pd.DataFrame, panel_name: str) -> Path:
    """Violin + box plot of EBM stage distribution across diagnosis groups."""
    stage_col = PANEL_STAGE_COLS[panel_name]
    label     = PANEL_LABELS[panel_name]
 
    dx_order = [config.DX_CN, config.DX_CIND, config.DX_AD]
    data = [cohort.loc[cohort["diagnosis_group"] == dx, stage_col].dropna().values
            for dx in dx_order]
 
    fig, ax = plt.subplots(figsize=(7, 5))
 
    vp = ax.violinplot(data, positions=range(len(dx_order)),
                       showmedians=True, showextrema=False)
    for i, (patch, dx) in enumerate(zip(vp["bodies"], dx_order)):
        patch.set_facecolor(_PALETTE.get(dx, "grey"))
        patch.set_alpha(0.6)
    vp["cmedians"].set_color("black")
    vp["cmedians"].set_linewidth(2)
 
    ax.boxplot(data, positions=range(len(dx_order)),
               widths=0.15, patch_artist=False,
               medianprops=dict(color="black", linewidth=0),
               whiskerprops=dict(linewidth=0.8),
               capprops=dict(linewidth=0.8),
               flierprops=dict(marker="o", markersize=2, alpha=0.4))
 
    ax.set_xticks(range(len(dx_order)))
    ax.set_xticklabels(dx_order, fontsize=11)
    ax.set_ylabel("EBM Stage", fontsize=11)
    ax.set_xlabel("Diagnosis Group", fontsize=11)
    ax.set_title(f"EBM Stage by Diagnosis — {label}",
                 fontsize=12, fontweight="bold")
    _apply_pub_style(ax)
    plt.tight_layout()
 
    out = _fig_path(f"fig03_stage_by_diagnosis_{panel_name}.png")
    fig.savefig(out, dpi=_DPI, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved: %s", out)
    return out
 
 
def fig_baseline_cognition_by_stage(
    cohort: pd.DataFrame, panel_name: str,
) -> Path:
    """Scatter + regression of stage vs baseline MMSE and CDR-SB."""
    stage_col = PANEL_STAGE_COLS[panel_name]
    label     = PANEL_LABELS[panel_name]
 
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
 
    for ax, (outcome_label, outcome_col) in zip(
        axes,
        [("Baseline MMSE", config.MMSE_COL), ("Baseline CDR-SB", config.CDR_SB_COL)],
    ):
        valid = cohort[[stage_col, outcome_col, "diagnosis_group"]].dropna()
        for dx in [config.DX_CN, config.DX_CIND, config.DX_AD]:
            sub = valid[valid["diagnosis_group"] == dx]
            jitter = np.random.default_rng(42).uniform(-0.15, 0.15, len(sub))
            ax.scatter(
                sub[stage_col] + jitter, sub[outcome_col],
                color=_PALETTE.get(dx, "grey"), alpha=0.35,
                s=18, label=dx, linewidths=0,
            )
        x = valid[stage_col].values.astype(float)
        y = valid[outcome_col].values.astype(float)
        slope, intercept, *_ = stats.linregress(x, y)
        xf = np.linspace(x.min(), x.max(), 100)
        ax.plot(xf, intercept + slope * xf, color="black", lw=1.5, ls="--", alpha=0.8)
 
        rho, p = stats.spearmanr(x, y)
        ax.set_xlabel("EBM Stage", fontsize=11)
        ax.set_ylabel(outcome_label, fontsize=11)
        ax.set_title(f"{outcome_label} vs Stage\n"
                     f"ρ={rho:.3f}, p{_format_p_star(p)}",
                     fontsize=11, fontweight="bold")
        _apply_pub_style(ax)
 
    axes[0].legend(title="Diagnosis", fontsize=8, title_fontsize=8)
    fig.suptitle(f"Baseline Cognition vs EBM Stage — {label}",
                 fontsize=13, fontweight="bold", y=1.01)
    plt.tight_layout()
 
    out = _fig_path(f"fig_baseline_cognition_{panel_name}.png")
    fig.savefig(out, dpi=_DPI, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved: %s", out)
    return out
 
 
def fig_predicted_trajectories(
    trajectories_df: pd.DataFrame,
    panel_name: str,
) -> Path:
    """Model-predicted cognitive trajectories stratified by EBM stage."""
    label = PANEL_LABELS[panel_name]
    sub   = trajectories_df[trajectories_df["panel"] == panel_name]
 
    if sub.empty:
        logger.warning("[%s] No trajectory data — skipping figure.", panel_name)
        return _fig_path(f"fig04_predicted_trajectories_{panel_name}_empty.png")
 
    outcomes = sub["outcome"].unique()
    fig, axes = plt.subplots(1, len(outcomes), figsize=(6 * len(outcomes), 5))
    if len(outcomes) == 1:
        axes = [axes]
 
    stage_colors = {"Stage 0": "#2166AC", "Stage 2": "#F4A582", "Stage 5": "#D6604D"}
    stage_colors_6 = {"Stage 0": "#2166AC", "Stage 3": "#F4A582", "Stage 6": "#D6604D"}
 
    for ax, outcome in zip(axes, outcomes):
        oc_sub = sub[sub["outcome"] == outcome]
        for stage_level in oc_sub["stage_level"].unique():
            sl_sub = oc_sub[oc_sub["stage_level"] == stage_level]
            sl_sub = sl_sub.sort_values("years")
            col    = stage_colors.get(stage_level,
                     stage_colors_6.get(stage_level, "grey"))
            ax.plot(sl_sub["years"], sl_sub["predicted"],
                    color=col, lw=2, label=stage_level)
 
        ax.set_xlabel("Years since baseline", fontsize=11)
        ax.set_ylabel(outcome, fontsize=11)
        ax.set_title(f"Predicted {outcome} Trajectories", fontsize=11, fontweight="bold")
        ax.legend(title="Baseline Stage", fontsize=9, title_fontsize=9)
        _apply_pub_style(ax)
 
    fig.suptitle(f"Model-Predicted Cognitive Decline — {label}",
                 fontsize=13, fontweight="bold", y=1.01)
    plt.tight_layout()
 
    out = _fig_path(f"fig04_predicted_trajectories_{panel_name}.png")
    fig.savefig(out, dpi=_DPI, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved: %s", out)
    return out
 
 
def fig_forest_plot(
    lme_results_df: pd.DataFrame,
    panel_name: str,
) -> Path:
    """Forest plot of mixed-effects model coefficients (stage terms only)."""
    label = PANEL_LABELS[panel_name]
    sub   = lme_results_df[
        (lme_results_df["panel"] == panel_name)
        & lme_results_df["parameter"].str.contains("stage", case=False, na=False)
    ].copy()
 
    if sub.empty:
        logger.warning("[%s] No LME results for forest plot — skipping.", panel_name)
        return _fig_path(f"fig05_forest_plot_{panel_name}_empty.png")
 
    fig, ax = plt.subplots(figsize=(8, max(3, len(sub) * 0.7)))
 
    y_pos = range(len(sub))
    colors = ["#D6604D" if sig else "#92C5DE"
              for sig in sub["significant"].values]
 
    ax.scatter(sub["beta"].values, list(y_pos), c=colors, s=60, zorder=5)
    for i, (_, row) in enumerate(sub.iterrows()):
        ax.hlines(i, row["ci_95_low"], row["ci_95_high"],
                  colors=colors[i], lw=2.5, alpha=0.7)
 
    ax.axvline(0, color="black", lw=0.8, ls="--", alpha=0.6)
    ax.set_yticks(list(y_pos))
    ax.set_yticklabels(
        [f"{r['parameter']}  ({r['outcome']})" for _, r in sub.iterrows()],
        fontsize=9,
    )
    ax.set_xlabel("Regression coefficient (β)", fontsize=11)
    ax.set_title(f"Mixed-Effects Model — Stage Coefficients\n{label}",
                 fontsize=12, fontweight="bold")
    _apply_pub_style(ax)
    plt.tight_layout()
 
    out = _fig_path(f"fig05_forest_plot_{panel_name}.png")
    fig.savefig(out, dpi=_DPI, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved: %s", out)
    return out
 
 
def fig_residual_diagnostics(
    fit_result: dict[str, Any] | None,
    panel_name: str,
    outcome_label: str,
) -> Path:
    """Residual diagnostic plots for model validation (Q-Q and fitted vs residuals)."""
    if fit_result is None:
        return _fig_path(f"fig_residuals_{panel_name}_{outcome_label}_empty.png")
 
    fit      = fit_result["fit"]
    resids   = fit.resid
 
    fig = plt.figure(figsize=(11, 4))
    gs  = gridspec.GridSpec(1, 2, figure=fig, wspace=0.3)
    ax1 = fig.add_subplot(gs[0])
    stats.probplot(resids, dist="norm", plot=ax1)
    ax1.set_title("Q-Q Plot of Residuals", fontsize=11, fontweight="bold")
    _apply_pub_style(ax1)
    ax2 = fig.add_subplot(gs[1])
    ax2.scatter(fit.fittedvalues, resids, alpha=0.3, s=12, color="#4393C3",
                linewidths=0)
    ax2.axhline(0, color="red", lw=1, ls="--")
    ax2.set_xlabel("Fitted values", fontsize=11)
    ax2.set_ylabel("Residuals", fontsize=11)
    ax2.set_title("Fitted vs Residuals", fontsize=11, fontweight="bold")
    _apply_pub_style(ax2)
 
    label = PANEL_LABELS[panel_name]
    fig.suptitle(f"Residual Diagnostics — {label} ({outcome_label})",
                 fontsize=12, fontweight="bold", y=1.02)
    plt.tight_layout()
 
    out = _fig_path(f"fig_residuals_{panel_name}_{outcome_label.replace(' ','_')}.png")
    fig.savefig(out, dpi=_DPI, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved: %s", out)
    return out

def build_validation_summary(
    cross_sectional: pd.DataFrame,
    sensitivity: pd.DataFrame,
    lme_rows: list[dict[str, Any]],
    conversion_rows: list[dict[str, Any]],
    panel_name: str,
) -> pd.DataFrame:
    """Produce a concise one-page validation summary for the panel.
 
    Combines cross-sectional, LME, and conversion results into a single
    structured DataFrame for reporting.
    """
    rows: list[dict[str, Any]] = []
 
    # Cross-sectional
    for _, r in cross_sectional.iterrows():
        if r.get("analysis") == "cross_sectional":
            rows.append({
                "panel": panel_name,
                "analysis": "cross_sectional",
                "outcome": r["outcome"],
                "n": r["n"],
                "statistic": f"ρ={r['spearman_rho']}",
                "p_value": r["p_value"],
                "note": f"95%CI [{r.get('rho_ci_95_low','?')}, {r.get('rho_ci_95_high','?')}]",
            })
    for r in lme_rows:
        if "stage:years" in r.get("parameter", "") or "stage × years" in r.get("parameter", ""):
            rows.append({
                "panel": panel_name,
                "analysis": "LME_interaction",
                "outcome": r["outcome"],
                "n": r["n_subjects"],
                "statistic": f"β={r['beta']}",
                "p_value": r["p_value"],
                "note": f"SE={r['SE']}, 95%CI [{r['ci_95_low']}, {r['ci_95_high']}]",
            })
    for _, r in sensitivity.iterrows():
        rows.append({
            "panel": panel_name,
            "analysis": f"sensitivity_{r.get('sensitivity_check','?')}",
            "outcome": r.get("outcome",""),
            "n": r.get("n",""),
            "statistic": f"ρ={r.get('spearman_rho','?')}",
            "p_value": r.get("p_value",""),
            "note": "",
        })
    for r in conversion_rows:
        if "stage" in r.get("parameter", "").lower():
            rows.append({
                "panel": panel_name,
                "analysis": f"logistic_{r['outcome']}",
                "outcome": r["outcome"],
                "n": r["n_total"],
                "statistic": f"OR={r['OR']}",
                "p_value": r["p_value"],
                "note": f"95%CI [{r['OR_ci_95_low']}, {r['OR_ci_95_high']}]",
            })
 
    return pd.DataFrame(rows)

def _format_p(p: float) -> str:
    """Format a p-value for publication (APA style)."""
    if np.isnan(p):
        return "NaN"
    if p < 0.001:
        return "<0.001"
    return f"{p:.3f}"
 
 
def _format_p_star(p: float) -> str:
    """Return significance stars for plot annotations."""
    if p < 0.001:
        return "<0.001"
    if p < 0.01:
        return "<0.01"
    if p < 0.05:
        return "<0.05"
    return f"={p:.3f}"
 
 
def _spearman_ci(rho: float, n: int, alpha: float = 0.05) -> tuple[float, float]:
    """95% CI for Spearman rho using Fisher z-transformation."""
    if n < 4:
        return (float("nan"), float("nan"))
    z     = np.arctanh(rho)
    se    = 1.0 / np.sqrt(n - 3)
    z_crit = stats.norm.ppf(1 - alpha / 2)
    lo    = float(np.tanh(z - z_crit * se))
    hi    = float(np.tanh(z + z_crit * se))
    return lo, hi
 
 
def _lme_rows_to_df(lme_results: dict[str, dict[str, Any] | None]) -> list[dict[str, Any]]:
    """Flatten a {key: fit_result} dict into a list of coefficient rows."""
    out: list[dict[str, Any]] = []
    for _, res in lme_results.items():
        if res is not None:
            out.extend(res.get("rows", []))
    return out

def run_panel_validation(
    baseline:      pd.DataFrame,
    visits:        pd.DataFrame | None,
    panel_name:    str,
    exclusion_log: list[dict[str, Any]],
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    list[dict],
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
]:
    """Execute the full validation pipeline for one panel.
 
    Returns all tabular results for later saving.
    """
    stage_col = PANEL_STAGE_COLS[panel_name]
    logger.info("=" * 60)
    logger.info("LONGITUDINAL VALIDATION — %s", panel_name.upper())
    logger.info("=" * 60)
    
    cohort = prepare_analysis_cohort(baseline, panel_name, exclusion_log)
    descriptives = compute_descriptive_stats(cohort, panel_name)
    cross_sec = cross_sectional_correlations(cohort, panel_name)
    fig_stage_histogram(cohort, panel_name)
    fig_stage_by_diagnosis(cohort, panel_name)
    fig_baseline_cognition_by_stage(cohort, panel_name)
    lme_results: dict[str, dict | None] = {}
    trajectories_list: list[pd.DataFrame] = []
 
    if visits is not None:
        long_df = build_longitudinal_long(visits, cohort, panel_name)
        n_stages = int(cohort[stage_col].max())
 
        for outcome_label, outcome_col in OUTCOMES.items():
            if outcome_col not in long_df.columns:
                continue
            res = fit_mixed_effects_model(
                long_df, outcome_col, stage_col, panel_name, outcome_label,
            )
            lme_results[f"{panel_name}_{outcome_label}"] = res
 
            traj = compute_predicted_trajectories(
                long_df, res, outcome_col, panel_name, outcome_label, n_stages,
            )
            trajectories_list.append(traj)
 
            fig_residual_diagnostics(res, panel_name, outcome_label)
 
        trajectories_df = pd.concat(trajectories_list, ignore_index=True) if trajectories_list else pd.DataFrame()
        fig_predicted_trajectories(trajectories_df, panel_name)
    else:
        logger.warning("[%s] Skipping LME — no visit data.", panel_name)
        trajectories_df = pd.DataFrame()
 
    lme_rows = _lme_rows_to_df(lme_results)

    if lme_rows:
        lme_df_tmp = pd.DataFrame(lme_rows)
        fig_forest_plot(lme_df_tmp, panel_name)
    conversion_df = conversion_analysis(cohort, panel_name)
    conversion_rows = conversion_df.to_dict(orient="records")
    sensitivity_df = sensitivity_analyses(cohort, visits, panel_name)
    summary = build_validation_summary(
        cross_sec, sensitivity_df, lme_rows, conversion_rows, panel_name,
    )
 
    return (
        descriptives,
        cross_sec,
        lme_rows,
        trajectories_df,
        conversion_df,
        sensitivity_df,
        summary,
    )

def save_results(
    all_descriptives: pd.DataFrame,
    all_cross_sec: pd.DataFrame,
    all_lme_rows: list[dict[str, Any]],
    all_trajectories: pd.DataFrame,
    all_conversion: pd.DataFrame,
    all_sensitivity: pd.DataFrame,
    all_summaries: pd.DataFrame,
    exclusion_log: list[dict[str, Any]],
    upstream_paths: list[Path],
) -> None:
    """Write all tabular outputs and the provenance metadata file."""
    config.ensure_project_dirs()
    config.LONGITUDINAL_RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    config.TABLES_DIR.mkdir(parents=True, exist_ok=True)
 
    save_csv(all_summaries,
             config.LONGITUDINAL_RESULTS_DIR / "validation_summary.csv")
 
    if all_lme_rows:
        save_csv(pd.DataFrame(all_lme_rows),
                 config.LONGITUDINAL_RESULTS_DIR / "mixed_effects_results.csv")
        save_csv(pd.DataFrame(all_lme_rows),
                 config.TABLES_DIR / "table04_mixed_model_results.csv")
 
    if not all_conversion.empty:
        save_csv(all_conversion,
                 config.LONGITUDINAL_RESULTS_DIR / "conversion_results.csv")
 
    if not all_trajectories.empty:
        save_csv(all_trajectories, config.PREDICTED_TRAJECTORIES_FILE)
 
    if not all_descriptives.empty:
        save_csv(all_descriptives,
                 config.TABLES_DIR / "table01_cohort_characteristics.csv")
 
    if not all_sensitivity.empty:
        save_csv(all_sensitivity,
                 config.LONGITUDINAL_RESULTS_DIR / "sensitivity_results.csv")
 
    meta = build_run_metadata("04_longitudinal_validation.py")
    meta.update({
        "project_version": config.PROJECT_VERSION,
        "panels_validated": list(PANEL_STAGE_COLS.keys()),
        "outcomes": list(OUTCOMES.keys()),
        "n_lme_results": len(all_lme_rows),
        "exclusion_log":  exclusion_log,
        "upstream_fingerprints": {
            str(p): file_fingerprint(p) if p.exists() else "MISSING"
            for p in upstream_paths
        },
    })
    write_metadata(
        meta,
        config.LONGITUDINAL_RESULTS_DIR / "longitudinal_metadata.json",
    )
 
    logger.info(
        "Results saved to %s and %s",
        config.LONGITUDINAL_RESULTS_DIR, config.TABLES_DIR,
    )

def main() -> None:
    """Orchestrate the longitudinal validation pipeline."""
    config.ensure_project_dirs()
 
    logger.info("=" * 70)
    logger.info("04_longitudinal_validation.py  |  version %s",
                config.PROJECT_VERSION)
    logger.info("=" * 70)
 
    exclusion_log: list[dict[str, Any]] = []
    baseline = load_baseline_dataset()
    visits   = load_longitudinal_visits()
    all_descriptives: list[pd.DataFrame] = []
    all_cross_sec: list[pd.DataFrame] = []
    all_lme_rows: list[dict] = []
    all_trajectories: list[pd.DataFrame] = []
    all_conversion: list[pd.DataFrame] = []
    all_sensitivity: list[pd.DataFrame] = []
    all_summaries: list[pd.DataFrame] = []

    for panel_name in [PANEL_MRI, PANEL_AMYLOID]:
        stage_col = PANEL_STAGE_COLS[panel_name]
        if stage_col not in baseline.columns:
            logger.warning("Stage column '%s' not found — skipping %s.",
                           stage_col, panel_name)
            continue
 
        (desc, cross, lme_rows, traj, conv, sens, summ) = run_panel_validation(
            baseline, visits, panel_name, exclusion_log,
        )
        all_descriptives.append(desc)
        all_cross_sec.append(cross)
        all_lme_rows.extend(lme_rows)
        all_trajectories.append(traj)
        all_conversion.append(conv)
        all_sensitivity.append(sens)
        all_summaries.append(summ)
    
    save_results(
        all_descriptives  = pd.concat(all_descriptives, ignore_index=True)  if all_descriptives  else pd.DataFrame(),
        all_cross_sec     = pd.concat(all_cross_sec,    ignore_index=True)  if all_cross_sec     else pd.DataFrame(),
        all_lme_rows      = all_lme_rows,
        all_trajectories  = pd.concat(all_trajectories, ignore_index=True)  if all_trajectories  else pd.DataFrame(),
        all_conversion    = pd.concat(all_conversion,   ignore_index=True)  if all_conversion    else pd.DataFrame(),
        all_sensitivity   = pd.concat(all_sensitivity,  ignore_index=True)  if all_sensitivity   else pd.DataFrame(),
        all_summaries     = pd.concat(all_summaries,    ignore_index=True)  if all_summaries     else pd.DataFrame(),
        exclusion_log     = exclusion_log,
        upstream_paths    = [
            config.PROCESSED_DIR / "longitudinal_validation_dataset.csv",
            config.LONGITUDINAL_RAW_FILE,
        ],
    )
 
    logger.info("=" * 70)
    logger.info("04_longitudinal_validation.py COMPLETE")
    logger.info("=" * 70)
 
 
if __name__ == "__main__":
    main()

















 

 
