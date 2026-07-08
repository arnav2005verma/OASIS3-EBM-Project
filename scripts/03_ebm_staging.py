from __future__ import annotations 
import argparse
import sys
import warnings
from collections import Counter
from dataclasses import dataclass, field
from itertools import permutations
from pathlib import Path
from typing import Any
import numpy as np
import pandas as pd
from scipy import stats
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
    save_pickle,
    write_metadata,
)
from utils.logging_utils import (
    append_exclusion_log,
    get_logger,
    log_dataframe_shape,
    log_cascading_filter,
    log_warning_file,
)
from utils.validation_utils import (
    ValidationError,
    assert_columns_present,
    assert_no_duplicate_id,
    assert_non_empty_dataframe,
    assert_value_within_range,
)
try:
    from kde_ebm.mixture_model import fit_all_gmm_models, get_prob_mat
    from kde_ebm.mcmc import mcmc as _kde_mcmc
except ImportError as exc:
    raise ImportError(
        "kde_ebm is required for 03_ebm_staging.py.\n"
        "Install with:  pip install git+https://github.com/ucl-pond/kde_ebm.git\n"
        f"Original error: {exc}"
    ) from exc
 
logger = get_logger(__name__, log_file=config.LOGS_DIR / "03_ebm_staging.log")

PANEL_MRI = "mri_only"
PANEL_AMYLOID = "mri_amyloid"
 
MRI_Z_COLS: list[str] = [
    "z_hippocampus_vol",
    "z_entorhinal_thickness",
    "z_fusiform_vol",
    "z_inferiortemporal_vol",
    "z_ventricular_vol",
]
 
AMYLOID_Z_COLS: list[str] = MRI_Z_COLS + ["z_Centiloid_fSUVR_TOT_CORTMEAN"]
 
# Columns passed through to every output file
_META_COLS: list[str] = [
    config.ID_COL,
    "diagnosis_group",
    "age_at_baseline",
    "sex",
    config.MMSE_COL,
    config.CDR_GLOBAL_COL,
    config.CDR_SB_COL,
    "adequate_followup",
]
 
_BURN_IN_FRACTION: float = 0.25

@dataclass
class PanelConfig:
    """Static description of one EBM panel."""
    panel_name: str
    csv_path: Path
    biomarker_z_cols: list[str]
    n_biomarkers: int = field(init=False)
 
    def __post_init__(self) -> None:
        self.n_biomarkers = len(self.biomarker_z_cols)
 
    def output_dir(self) -> Path:
        return config.STAGING_RESULTS_DIR / self.panel_name
 
 
@dataclass
class PanelResult:
    """All artefacts produced by running one EBM panel."""
    panel_name: str
    panel_df: pd.DataFrame
    subject_ids: np.ndarray
    biomarker_names: list[str]
    X: np.ndarray
    y: np.ndarray
    mixture_models: list
    mixture_quality: dict[str, dict]
    ml_sequence: np.ndarray 
    ml_loglikelihood: float
    mcmc_samples: np.ndarray
    bootstrap_sequences: np.ndarray | None
    pvd_matrix: np.ndarray | None
    stages_df: pd.DataFrame
    permutation_result: dict
    face_validity: dict
    converged: bool

def load_panel(panel_cfg: PanelConfig) -> pd.DataFrame:
    """Load and structurally validate one panel CSV.
 
    Raises
    ------
    FileNotFoundError  If the CSV is missing.
    ValidationError    If required columns are absent or IDs duplicated.
    """
    path = panel_cfg.csv_path
    if not path.exists():
        raise FileNotFoundError(
            f"Panel dataset not found: {path}\n"
            "Run 02_feature_engineering.py first."
        )
    df = pd.read_csv(path, low_memory=False)
    log_dataframe_shape(logger, df, f"panel={panel_cfg.panel_name}")
 
    required = [config.ID_COL, "diagnosis_group"] + panel_cfg.biomarker_z_cols
    assert_columns_present(df, required)
    assert_no_duplicate_id(df, config.ID_COL)
    assert_non_empty_dataframe(df, panel_cfg.panel_name)
    return df
 
 
def build_ebm_arrays(
    df: pd.DataFrame,
    biomarker_cols: list[str],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Produce the X matrix, binary y vector, and subject-ID array.
 
    y = 0 for CN subjects, 1 for CIND or AD.
 
    Returns
    -------
    X           float64 (n_subjects, n_biomarkers)
    y           int32   (n_subjects,)
    subject_ids object  (n_subjects,) — OASISID strings
    """
    X = df[biomarker_cols].to_numpy(dtype=np.float64)
    y = (df["diagnosis_group"] != config.DX_CN).astype(np.int32).to_numpy()
    subject_ids = df[config.ID_COL].to_numpy()
 
    n_cn = int((y == 0).sum())
    n_non_cn = int((y == 1).sum())
    logger.info(
        "EBM arrays: %d subjects  (CN=%d, non-CN=%d)",
        len(y), n_cn, n_non_cn,
    )
 
    if n_cn < config.MIN_CN_REFERENCE_N:
        raise ValidationError(
            f"Only {n_cn} CN subjects — minimum required: {config.MIN_CN_REFERENCE_N}"
        )
    if n_non_cn < config.MIN_N_FOR_EBM:
        raise ValidationError(
            f"Only {n_non_cn} non-CN subjects — minimum required for EBM: "
            f"{config.MIN_N_FOR_EBM}"
        )
        for j, col in enumerate(biomarker_cols):
        col_data = X[:, j]
        if np.isnan(col_data).all():
            raise ValidationError(f"Biomarker '{col}' is all-NaN in the panel.")
        if col_data[~np.isnan(col_data)].std() == 0:
            raise ValidationError(f"Biomarker '{col}' has zero variance.")
 
    return X, y, subject_ids

def fit_mixture_models(
    X: np.ndarray,
    y: np.ndarray,
    random_seed: int,
) -> list:
    """Fit per-biomarker GMM mixture models using CN subjects as normal reference.
 
    All z-scores have "higher = more abnormal", so patholog_dirn=[1]*N.
 
    Args
    ----
    X            (n_subjects, N) z-score matrix.
    y            Binary vector (0=CN, 1=non-CN).
    random_seed  Set before fitting for reproducibility.
 
    Returns
    -------
    List of N fitted kde_ebm mixture-model objects.
    """
    np.random.seed(random_seed)
    N = X.shape[1]
    logger.info("Fitting %d GMM mixture models (seed=%d) …", N, random_seed)
    mixture_models = fit_all_gmm_models(
        X, y,
        fit_all_subjects=False,
        implement_fixed_controls=False,
        patholog_dirn=[1] * N,
    )
    logger.info("Mixture model fitting complete.")
    return mixture_models
 
 
def validate_mixture_quality(
    mixture_models: list,
    biomarker_names: list[str],
) -> dict[str, dict]:
    """Check separation between normal and abnormal mixture components.
 
    A biomarker whose components are nearly identical will not contribute
    meaningful signal to the EBM.  This function logs a WARNING for each
    such biomarker but does not halt execution.
 
    Returns
    -------
    dict  {biomarker_name: {normal_mean, abnormal_mean, separation, warning}}
    """
    quality: dict[str, dict] = {}
    _SEP_THRESHOLD = 0.5 
    for name, mm in zip(biomarker_names, mixture_models):
        try:
            params = {}
            for attr in ("means_", "means", "params"):
                if hasattr(mm, attr):
                    params["means"] = getattr(mm, attr)
                    break
            sep = float("nan")
            warn = False
            rec = {
                "normal_mean":   float("nan"),
                "abnormal_mean": float("nan"),
                "separation":    sep,
                "warning":       warn,
            }
        except Exception:
            rec = {"normal_mean": float("nan"), "abnormal_mean": float("nan"),
                   "separation": float("nan"), "warning": True}
 
        quality[name] = rec
    N = len(mixture_models)
    grid = np.linspace(-4, 4, 200).reshape(-1, 1)
    grid_full = np.tile(grid, (1, N))
 
    try:
        pm_grid = get_prob_mat(grid_full, mixture_models)
        for i, name in enumerate(biomarker_names):
            p_norm   = pm_grid[:, i, 0]
            p_abnorm = pm_grid[:, i, 1]
            norm_mean = float(np.average(grid.squeeze(), weights=p_norm))
            abnorm_mean = float(np.average(grid.squeeze(), weights=p_abnorm))
            sep = float(abs(abnorm_mean - norm_mean))
            warn = sep < _SEP_THRESHOLD
 
            quality[name] = {
                "normal_mean": round(norm_mean, 4),
                "abnormal_mean": round(abnorm_mean, 4),
                "separation": round(sep, 4),
                "warning": warn,
            }
            level = logger.warning if warn else logger.info
            level(
                "Mixture quality '%s': normal_mean=%.3f  abnormal_mean=%.3f  "
                "separation=%.3f%s",
                name, norm_mean, abnorm_mean, sep,
                "  *** LOW SEPARATION — treat ordering with caution ***" if warn else "",
            )
    except Exception as exc:
        logger.warning("Could not evaluate mixture quality on grid: %s", exc)
 
    return quality

def _sequence_loglik(prob_mat: np.ndarray, sequence: np.ndarray | list) -> float:
    """Dataset log-likelihood for one biomarker ordering.
 
    Parameters
    ----------
    prob_mat  (n_subjects, N, 2)  [:,:,0]=P_normal  [:,:,1]=P_abnormal
    sequence  (N,)  biomarker indices in event order (0 = first abnormal)
 
    Returns
    -------
    Scalar float — sum of log marginal likelihoods across subjects.
    """
    seq = np.asarray(sequence, dtype=int)
    n_subjects, N = prob_mat.shape[:2]
 
    log_pn  = np.log(np.clip(prob_mat[:, :, 0], 1e-12, 1.0))
    log_pa  = np.log(np.clip(prob_mat[:, :, 1], 1e-12, 1.0))
    log_pn_ord  = log_pn[:, seq] 
    log_pa_ord  = log_pa[:, seq]
    total_log_n  = log_pn_ord.sum(axis=1)
    cum_log_pa   = np.cumsum(log_pa_ord, axis=1)
    cum_log_pn   = np.cumsum(log_pn_ord, axis=1)
    stage_ll = np.empty((n_subjects, N + 1))
    stage_ll[:, 0] = total_log_n
    for k in range(1, N + 1):
        stage_ll[:, k] = cum_log_pa[:, k - 1] + total_log_n - cum_log_pn[:, k - 1]
    max_ll  = stage_ll.max(axis=1, keepdims=True)
    log_sum = max_ll.squeeze() + np.log(
        np.exp(stage_ll - max_ll).mean(axis=1)
    )
    return float(log_sum.sum())
 
 
def _compute_stage_posteriors(
    prob_mat: np.ndarray,
    sequence: np.ndarray,
) -> np.ndarray:
    """Posterior P(stage=k | x_i, sigma) for all subjects and stages 0..N.
 
    Parameters
    ----------
    prob_mat   (n_subjects, N, 2)
    sequence   (N,)  biomarker indices in event order
 
    Returns
    -------
    posteriors  (n_subjects, N+1)  rows sum to 1
    """
    seq        = np.asarray(sequence, dtype=int)
    n, N       = prob_mat.shape[:2]
    log_pn     = np.log(np.clip(prob_mat[:, :, 0], 1e-12, 1.0))
    log_pa     = np.log(np.clip(prob_mat[:, :, 1], 1e-12, 1.0))
    log_pn_ord = log_pn[:, seq]
    log_pa_ord = log_pa[:, seq]
 
    total_log_n = log_pn_ord.sum(axis=1)
    cum_log_pa  = np.cumsum(log_pa_ord, axis=1)
    cum_log_pn  = np.cumsum(log_pn_ord, axis=1)
 
    stage_ll = np.empty((n, N + 1))
    stage_ll[:, 0] = total_log_n
    for k in range(1, N + 1):
        stage_ll[:, k] = cum_log_pa[:, k - 1] + total_log_n - cum_log_pn[:, k - 1]
 
    max_ll     = stage_ll.max(axis=1, keepdims=True)
    exp_ll     = np.exp(stage_ll - max_ll)
    posteriors = exp_ll / exp_ll.sum(axis=1, keepdims=True)
    return posteriors

def run_mcmc(
    X: np.ndarray,
    mixture_models: list,
    n_iter: int,
    n_greedy_iter: int,
    n_greedy_init: int,
    random_seed: int,
) -> tuple[np.ndarray, float, np.ndarray]:
    """Run MCMC sequence optimisation via kde_ebm.
 
    kde_ebm.mcmc returns a list of EventOrder(order, score) objects, one per
    MCMC iteration.  This function extracts:
        ml_sequence      — order of the highest-score EventOrder   (N,)
        ml_loglikelihood — its score
        samples_array    — stacked orders from all iterations       (n_iter, N)
 
    Args
    ----
    random_seed  Set np.random.seed before calling kde_ebm (it uses global state).
 
    Returns
    -------
    (ml_sequence, ml_loglikelihood, samples_array)
    """
    np.random.seed(random_seed)
    N = X.shape[1]
    logger.info(
        "MCMC: n_iter=%d  greedy_iter=%d  greedy_init=%d  seed=%d",
        n_iter, n_greedy_iter, n_greedy_init, random_seed,
    )
 
    event_orders = _kde_mcmc(
        X, mixture_models,
        n_iter=n_iter,
        greedy_n_iter=n_greedy_iter,
        greedy_n_init=n_greedy_init,
        plot=False,
    )
    # event_orders: list of EventOrder(order=ndarray, score=float), len=n_iter
    ml_event = max(event_orders, key=lambda e: e.score)
    ml_sequence = np.asarray(ml_event.order, dtype=int)
    ml_loglikelihood = float(ml_event.score)
 
    samples_array = np.array([e.order for e in event_orders], dtype=int)  # (n_iter, N)
 
    logger.info(
        "MCMC complete: ML log-likelihood=%.4f  ML sequence=%s",
        ml_loglikelihood,
        ml_sequence.tolist(),
    )
    return ml_sequence, ml_loglikelihood, samples_array
 
 
def check_mcmc_convergence(
    samples: np.ndarray,
    burn_in_fraction: float = _BURN_IN_FRACTION,
) -> dict[str, Any]:
    """Compare modal sequence in the early and late halves of the MCMC trace.
 
    After discarding the burn-in, the trace is split at the midpoint.  If
    both halves agree on the modal sequence the chain is declared converged.
 
    Returns
    -------
    dict with keys: converged, early_half_mode, late_half_mode, match
    """
    n_iter = len(samples)
    burn_in = int(n_iter * burn_in_fraction)
    post_burnin = samples[burn_in:]
    mid = len(post_burnin) // 2
    early = post_burnin[:mid]
    late  = post_burnin[mid:]
 
    def _modal(arr: np.ndarray) -> tuple:
        return Counter(map(tuple, arr)).most_common(1)[0][0]
 
    early_mode = _modal(early)
    late_mode  = _modal(late)
    match      = early_mode == late_mode
 
    result = {
        "converged": match,
        "early_half_mode": list(early_mode),
        "late_half_mode": list(late_mode),
        "match": match,
        "burn_in_fraction": burn_in_fraction,
        "n_post_burnin": len(post_burnin),
    }
 
    if match:
        logger.info("MCMC convergence check: PASSED — early and late modes agree.")
    else:
        logger.warning(
            "MCMC convergence check: WARNING — modes differ.  "
            "early=%s  late=%s  Consider increasing N_MCMC_ITER.",
            list(early_mode), list(late_mode),
        )
    return result

def assign_stages(
    X: np.ndarray,
    mixture_models: list,
    ml_sequence: np.ndarray,
    subject_ids: np.ndarray,
    panel_df: pd.DataFrame,
    biomarker_names: list[str],
) -> pd.DataFrame:
    """Compute MAP stage and posterior entropy for every subject.
 
    Stage k means biomarkers in positions 0..k-1 of the ML sequence have
    become abnormal for that subject.
 
    Columns in returned DataFrame
    -----------------------------
    OASISID, diagnosis_group, stage, stage_posterior_entropy,
    stage_prob_0, stage_prob_1, …, stage_prob_N
    """
    logger.info("Assigning stages …")
    prob_mat = get_prob_mat(X, mixture_models)
    posteriors = _compute_stage_posteriors(prob_mat, ml_sequence)
    N = len(ml_sequence)
    map_stages = np.argmax(posteriors, axis=1).astype(int)
    eps = 1e-12
    entropy = -np.sum(posteriors * np.log2(posteriors + eps), axis=1)
 
    result = pd.DataFrame({config.ID_COL: subject_ids})
    result["diagnosis_group"] = panel_df["diagnosis_group"].values
    result["stage"] = map_stages
    result["stage_posterior_entropy"] = np.round(entropy, 4)
 
    for k in range(N + 1):
        result[f"stage_prob_{k}"] = np.round(posteriors[:, k], 6)
        assert_value_within_range(result["stage"], min_value=0, max_value=N)
 
    logger.info(
        "Stage assignment complete: mean=%.2f  median=%.1f  "
        "stage_0_n=%d  stage_N_n=%d",
        map_stages.mean(),
        float(np.median(map_stages)),
        int((map_stages == 0).sum()),
        int((map_stages == N).sum()),
    )
    for dx in [config.DX_CN, config.DX_CIND, config.DX_AD]:
        mask = result["diagnosis_group"] == dx
        if mask.any():
            logger.info(
                "  Mean stage  %s = %.2f  (n=%d)",
                dx, result.loc[mask, "stage"].mean(), int(mask.sum()),
            )
 
    return result

def run_permutation_test(
    X: np.ndarray,
    mixture_models: list,
    ml_sequence: np.ndarray,
    ml_loglikelihood: float,
    n_permutations: int,
    random_seed: int,
) -> dict[str, Any]:
    """Assess whether the ML sequence is significantly better than chance.
 
    For each permutation:
      1. Independently shuffle each biomarker column (breaks stage association).
      2. Compute the best possible log-likelihood on shuffled data under the
         FIXED mixture models by exhaustive enumeration (N ≤ 6 → max 720 seqs).
      3. Compare null distribution to the observed ML log-likelihood.
 
    Returns
    -------
    dict with observed_loglik, null_mean, null_sd, p_value, n_permutations
    """
    rng = np.random.default_rng(random_seed)
    N = X.shape[1]
    logger.info("Permutation test: n_permutations=%d …", n_permutations)
 
    all_seqs = list(permutations(range(N)))
    null_ll: list[float] = []
    for _ in range(n_permutations):
        X_perm = X.copy()
        for j in range(N):
            idx = rng.permutation(X.shape[0])
            X_perm[:, j] = X[idx, j]
 
        pm_perm = get_prob_mat(X_perm, mixture_models)
        best    = max(_sequence_loglik(pm_perm, list(s)) for s in all_seqs)
        null_ll.append(best)
 
    null_arr = np.array(null_ll)
    p_value  = float((null_arr >= ml_loglikelihood).mean())
 
    result = {
        "observed_loglik": round(ml_loglikelihood, 4),
        "null_mean":       round(float(null_arr.mean()), 4),
        "null_sd":         round(float(null_arr.std()),  4),
        "p_value":         round(p_value, 4),
        "n_permutations":  n_permutations,
    }
    logger.info(
        "Permutation test: observed=%.4f  null_mean=%.4f  null_sd=%.4f  p=%.4f",
        ml_loglikelihood, result["null_mean"], result["null_sd"], p_value,
    )
    if p_value > 0.05:
        logger.warning(
            "Permutation p-value > 0.05 (p=%.3f): event sequence may not be "
            "significantly better than chance.", p_value,
        )
    return result

def run_bootstrap(
    X: np.ndarray,
    y: np.ndarray,
    n_bootstrap: int,
    n_iter: int,
    random_seed: int,
) -> np.ndarray:
    """Bootstrap resampling for positional variance diagrams.
 
    For each resample:
      1. Draw rows with replacement (stratified by y for stable CN reference).
      2. Re-fit GMM mixture models on the CN subjects in that resample.
      3. Run MCMC and record the ML sequence.
 
    Returns
    -------
    bootstrap_sequences  (n_bootstrap, N)  int array
    """
    rng = np.random.default_rng(random_seed)
    N   = X.shape[1]
    n   = X.shape[0]
    bootstrap_sequences = np.empty((n_bootstrap, N), dtype=int)
 
    logger.info("Bootstrap: %d resamples × %d MCMC iters …", n_bootstrap, n_iter)
 
    for r in range(n_bootstrap):
        resample_seed = int(random_seed + r + 1)
        idx = rng.integers(0, n, size=n)
        X_r, y_r = X[idx], y[idx]
        try:
            np.random.seed(resample_seed)
            mm_r = fit_all_gmm_models(
                X_r, y_r,
                fit_all_subjects=False,
                implement_fixed_controls=False,
                patholog_dirn=[1] * N,
            )
            np.random.seed(resample_seed)
            event_orders_r = _kde_mcmc(
                X_r, mm_r,
                n_iter=n_iter,
                greedy_n_iter=max(100, n_iter // 10),
                greedy_n_init=5,
                plot=False,
            )
            ml_r = max(event_orders_r, key=lambda e: e.score)
            bootstrap_sequences[r] = np.asarray(ml_r.order, dtype=int)
        except Exception as exc:
            logger.warning(
                "Bootstrap resample %d failed (%s); using ML sequence as fallback.",
                r, exc,
            )
            bootstrap_sequences[r] = bootstrap_sequences[0] if r > 0 else np.arange(N)
 
        if (r + 1) % max(1, n_bootstrap // 5) == 0:
            logger.info("  Bootstrap progress: %d / %d", r + 1, n_bootstrap)
 
    logger.info("Bootstrap complete.")
    return bootstrap_sequences
 
 
def compute_pvd(bootstrap_sequences: np.ndarray) -> np.ndarray:
    """Convert raw bootstrap sequences to a positional variance diagram.
 
    pvd[position i, biomarker j] = proportion of resamples where biomarker j
    was placed at sequence position i.  Each column sums to 1.
 
    Returns
    -------
    pvd  (N, N) float array
    """
    n_bootstrap, N = bootstrap_sequences.shape
    pvd = np.zeros((N, N), dtype=float)
    for r in range(n_bootstrap):
        for pos, bio_idx in enumerate(bootstrap_sequences[r]):
            pvd[pos, bio_idx] += 1.0
    pvd /= n_bootstrap
    return pvd

def validate_face_validity(
    stages_df: pd.DataFrame,
    panel_df: pd.DataFrame,
) -> dict[str, Any]:
    """Test whether EBM stage correlates sensibly with CDR and diagnosis.
 
    Scientific gate: a negative Spearman correlation between stage and CDR
    indicates a sign-convention error or a fundamentally wrong sequence and
    HALTS the pipeline (raises ValidationError).
 
    Returns
    -------
    dict with spearman_rho_cdr, spearman_p_cdr, kruskal_h, kruskal_p,
             mean_stage_by_dx
    """
    merged = stages_df.merge(
        panel_df[[config.ID_COL, config.CDR_GLOBAL_COL, "diagnosis_group"]],
        on=config.ID_COL, how="left",
    )
    valid = merged.dropna(subset=["stage", config.CDR_GLOBAL_COL])
 
    rho, p_rho = stats.spearmanr(valid["stage"], valid[config.CDR_GLOBAL_COL])
    rho, p_rho = float(rho), float(p_rho)
 
    logger.info(
        "Face validity — Spearman(stage, CDR): rho=%.4f  p=%.4g", rho, p_rho,
    )
 
    if rho < 0:
        raise ValidationError(
            f"FACE VALIDITY FAILED: Spearman rho between EBM stage and CDR is "
            f"negative ({rho:.4f}).  This indicates a sign-convention error "
            "or fundamentally wrong event sequence.  Do NOT proceed to "
            "longitudinal validation until this is resolved."
        )
    
    groups  = [
        merged.loc[merged["diagnosis_group"] == dx, "stage"].dropna().values
        for dx in [config.DX_CN, config.DX_CIND, config.DX_AD]
        if (merged["diagnosis_group"] == dx).any()
    ]
    if len(groups) >= 2 and all(len(g) > 0 for g in groups):
        h_stat, p_kruskal = stats.kruskal(*groups)
    else:
        h_stat, p_kruskal = float("nan"), float("nan")
 
    logger.info(
        "Face validity — Kruskal-Wallis (stage by Dx): H=%.4f  p=%.4g",
        h_stat, p_kruskal,
    )
 
    mean_stage_by_dx: dict[str, float] = {}
    for dx in [config.DX_CN, config.DX_CIND, config.DX_AD]:
        mask = merged["diagnosis_group"] == dx
        if mask.any():
            mean_val = float(merged.loc[mask, "stage"].mean())
            mean_stage_by_dx[dx] = round(mean_val, 4)
            logger.info("  Mean stage  %s = %.4f", dx, mean_val)
 
    return {
        "spearman_rho_cdr": round(rho,     4),
        "spearman_p_cdr":   round(p_rho,   6),
        "kruskal_h":        round(h_stat,  4),
        "kruskal_p":        round(p_kruskal, 6),
        "mean_stage_by_dx": mean_stage_by_dx,
    }

def _bootstrap_cache_path(panel_name: str) -> Path:
    return config.STAGING_RESULTS_DIR / panel_name / "bootstrap_sequences.npy"
 
 
def _bootstrap_meta_path(panel_name: str) -> Path:
    return config.STAGING_RESULTS_DIR / panel_name / "bootstrap_cache_meta.json"
 
 
def _load_bootstrap_cache(
    panel_name: str,
    n_bootstrap: int,
    n_iter: int,
    seed: int,
    n_subjects: int,
) -> np.ndarray | None:
    """Return cached bootstrap sequences if they match the current config."""
    bsp = _bootstrap_cache_path(panel_name)
    bmp = _bootstrap_meta_path(panel_name)
    if not bsp.exists() or not bmp.exists():
        return None
    try:
        from utils.io_utils import load_json
        meta = load_json(bmp)
        if (
            meta.get("n_bootstrap") == n_bootstrap
            and meta.get("n_mcmc_iter") == n_iter
            and meta.get("random_seed") == seed
            and meta.get("n_subjects") == n_subjects
        ):
            seqs = np.load(bsp)
            logger.info(
                "Loaded bootstrap cache for '%s' (%d resamples).",
                panel_name, n_bootstrap,
            )
            return seqs
    except Exception:
        pass
    return None
 
 
def _save_bootstrap_cache(
    panel_name: str,
    sequences: np.ndarray,
    n_bootstrap: int,
    n_iter: int,
    seed: int,
    n_subjects: int,
) -> None:
    bsp  = _bootstrap_cache_path(panel_name)
    bmp  = _bootstrap_meta_path(panel_name)
    bsp.parent.mkdir(parents=True, exist_ok=True)
    np.save(bsp, sequences)
    save_json(
        {"n_bootstrap": n_bootstrap, "n_mcmc_iter": n_iter,
         "random_seed": seed, "n_subjects": n_subjects},
        bmp,
    )

def run_panel(
    panel_cfg: PanelConfig,
    n_mcmc_iter: int,
    n_bootstrap: int,
    n_permutation: int,
    skip_bootstrap: bool,
    exclusion_log: list[dict[str, Any]],
) -> PanelResult:
    """Execute the full EBM workflow for one panel.
 
    Calls every Layer-2 function in sequence and wraps the results in a
    PanelResult dataclass.  Each panel is completely independent.
 
    Args
    ----
    panel_cfg       Static panel description (name, path, biomarker columns).
    n_mcmc_iter     MCMC iterations for primary run and bootstrap resamples.
    n_bootstrap     Number of bootstrap resamples for the PVD.
    n_permutation   Permutation test replications.
    skip_bootstrap  If True, skip bootstrap (PVD = None).
    exclusion_log   In-memory list appended in-place.
 
    Returns
    -------
    PanelResult
    """
    logger.info("=" * 60)
    logger.info("PANEL: %s", panel_cfg.panel_name.upper())
    logger.info("=" * 60)
 
    seed = config.RANDOM_SEED
    N = panel_cfg.n_biomarkers
    bio_cols = panel_cfg.biomarker_z_cols
    bio_names = [c.replace("z_", "") for c in bio_cols]

    panel_df = load_panel(panel_cfg)
    n_before = len(panel_df)
    X, y, subject_ids = build_ebm_arrays(panel_df, bio_cols)
    append_exclusion_log(exclusion_log, f"{panel_cfg.panel_name}_ebm_input",
                         n_before, len(panel_df))
    mixture_models  = fit_mixture_models(X, y, random_seed=seed)
    mixture_quality = validate_mixture_quality(mixture_models, bio_names)
    ml_sequence, ml_loglik, mcmc_samples = run_mcmc(
        X, mixture_models,
        n_iter=n_mcmc_iter,
        n_greedy_iter=max(1000, n_mcmc_iter // 10),
        n_greedy_init=10,
        random_seed=seed,
    )
    ml_biomarker_order = [bio_names[i] for i in ml_sequence]
    logger.info("ML biomarker event order: %s", " → ".join(ml_biomarker_order))
    convergence = check_mcmc_convergence(mcmc_samples)
    converged   = convergence["converged"]
    stages_df = assign_stages(
        X, mixture_models, ml_sequence, subject_ids, panel_df, bio_names,
    )
    prob_mat = get_prob_mat(X, mixture_models)
    permutation_result = run_permutation_test(
        X, mixture_models, ml_sequence, ml_loglik,
        n_permutations=n_permutation,
        random_seed=seed,
    )
    bootstrap_sequences: np.ndarray | None = None
    pvd_matrix:          np.ndarray | None = None
 
    if skip_bootstrap:
        logger.info("Bootstrap skipped (--skip-bootstrap).  PVD will be None.")
    else:
        cached = _load_bootstrap_cache(
            panel_cfg.panel_name, n_bootstrap, n_mcmc_iter, seed, len(panel_df)
        )
        if cached is not None:
            bootstrap_sequences = cached
        else:
            bootstrap_sequences = run_bootstrap(
                X, y,
                n_bootstrap=n_bootstrap,
                n_iter=n_mcmc_iter,
                random_seed=seed,
            )
            _save_bootstrap_cache(
                panel_cfg.panel_name, bootstrap_sequences,
                n_bootstrap, n_mcmc_iter, seed, len(panel_df),
            )
        pvd_matrix = compute_pvd(bootstrap_sequences)
    face_validity = validate_face_validity(stages_df, panel_df)
 
    return PanelResult(
        panel_name = panel_cfg.panel_name,
        panel_df = panel_df,
        subject_ids = subject_ids,
        biomarker_names = bio_names,
        X = X,
        y = y,
        mixture_models = mixture_models,
        mixture_quality = mixture_quality,
        ml_sequence = ml_sequence,
        ml_loglikelihood = ml_loglik,
        mcmc_samples = mcmc_samples,
        bootstrap_sequences = bootstrap_sequences,
        pvd_matrix = pvd_matrix,
        stages_df = stages_df,
        permutation_result = permutation_result,
        face_validity = face_validity,
        converged = converged,
    )

def merge_stages_to_longitudinal(
    longitudinal_df: pd.DataFrame,
    mri_result: PanelResult,
    amyloid_result: PanelResult,
) -> pd.DataFrame:
    """Fill the ebm_stage_mri and ebm_stage_mri_amyloid placeholder columns.
 
    Only subjects present in the respective panel are assigned a stage;
    subjects absent from a panel retain NaN (e.g., subjects without amyloid
    PET retain NaN in ebm_stage_mri_amyloid).
 
    Returns
    -------
    Updated longitudinal_validation_dataset DataFrame.
    """
    df = longitudinal_df.copy()
 
    for col, result in [
        ("ebm_stage_mri", mri_result),
        ("ebm_stage_mri_amyloid", amyloid_result),
    ]:
        stage_map = result.stages_df.set_index(config.ID_COL)["stage"]
        df[col] = df[config.ID_COL].map(stage_map)
        n_filled  = df[col].notna().sum()
        logger.info(
            "Merged '%s': %d / %d subjects received a stage.",
            col, n_filled, len(df),
        )
 
    return df
 
def build_cross_panel_summary(
    mri_result: PanelResult,
    amyloid_result: PanelResult,
) -> pd.DataFrame:
    """One row per biomarker per panel: position, confidence, permutation p.
 
    Positional confidence = proportion of bootstrap resamples placing the
    biomarker in its modal position (NaN if bootstrap was skipped).
    """
    rows: list[dict[str, Any]] = []
 
    for result in [mri_result, amyloid_result]:
        N = result.X.shape[1]
        seq = result.ml_sequence
        pvd = result.pvd_matrix
 
        for pos, bio_idx in enumerate(seq):
            conf = (
                float(pvd[pos, bio_idx])
                if pvd is not None else float("nan")
            )
            rows.append({
                "panel": result.panel_name,
                "sequence_position": pos + 1,  
                "biomarker": result.biomarker_names[bio_idx],
                "biomarker_col_index": int(bio_idx),
                "positional_confidence": round(conf, 4),
                "permutation_p": result.permutation_result.get("p_value"),
                "ml_loglikelihood": round(result.ml_loglikelihood, 4),
                "face_validity_rho_cdr": result.face_validity.get("spearman_rho_cdr"),
                "converged": result.converged,
            })
 
    return pd.DataFrame(rows)

def _panel_output_dir(panel_name: str) -> Path:
    d = config.STAGING_RESULTS_DIR / panel_name
    d.mkdir(parents=True, exist_ok=True)
    return d
 
 
def save_panel_outputs(result: PanelResult, panel_csv_path: Path) -> None:
    """Write all per-panel artefacts.
 
    Files
    -----
    event_sequence.csv — human-readable ML sequence with PVD confidence
    subject_stages.csv — per-subject stage and posterior
    mixture_models.pkl — fitted kde_ebm mixture model objects
    mcmc_samples.npy — (n_iter, N) MCMC trace
    bootstrap_sequences.npy — (n_bootstrap, N) or absent if skipped
    pvd_matrix.npy — (N, N) positional variance diagram or absent
    permutation_test.json
    face_validity_results.json
    staging_metadata.json
    """
    out = _panel_output_dir(result.panel_name)
    N   = len(result.ml_sequence)
    seq_rows = []
    for pos, bio_idx in enumerate(result.ml_sequence):
        conf = (
            float(result.pvd_matrix[pos, bio_idx])
            if result.pvd_matrix is not None else float("nan")
        )
        seq_rows.append({
            "position": pos + 1,
            "biomarker": result.biomarker_names[bio_idx],
            "biomarker_col_index": int(bio_idx),
            "positional_confidence": round(conf, 4),
        })
    save_csv(pd.DataFrame(seq_rows), out / "event_sequence.csv")
    save_csv(result.stages_df, out / "subject_stages.csv")
    save_pickle(result.mixture_models, out / "mixture_models.pkl")
    np.save(str(out / "mcmc_samples.npy"), result.mcmc_samples)
    if result.bootstrap_sequences is not None:
        np.save(str(out / "bootstrap_sequences.npy"), result.bootstrap_sequences)
    if result.pvd_matrix is not None:
        np.save(str(out / "pvd_matrix.npy"), result.pvd_matrix)
    save_json(result.permutation_result, out / "permutation_test.json")
    save_json(result.face_validity,      out / "face_validity_results.json")

    meta = build_run_metadata("03_ebm_staging.py")
    meta.update({
        "project_version": config.PROJECT_VERSION,
        "panel_name": result.panel_name,
        "n_subjects": int(result.X.shape[0]),
        "n_biomarkers": int(result.X.shape[1]),
        "biomarker_names": result.biomarker_names,
        "random_seed": config.RANDOM_SEED,
        "n_mcmc_iter": result.mcmc_samples.shape[0],
        "n_bootstrap": (
            result.bootstrap_sequences.shape[0]
            if result.bootstrap_sequences is not None else 0
        ),
        "bootstrap_skipped": result.bootstrap_sequences is None,
        "n_cn": int((result.y == 0).sum()),
        "n_non_cn": int((result.y == 1).sum()),
        "ml_sequence": result.ml_sequence.tolist(),
        "ml_biomarker_order": [result.biomarker_names[i] for i in result.ml_sequence],
        "ml_loglikelihood": round(result.ml_loglikelihood, 4),
        "converged": result.converged,
        "permutation_result": result.permutation_result,
        "face_validity": result.face_validity,
        "mixture_quality": result.mixture_quality,
        "source_fingerprint": (
            file_fingerprint(panel_csv_path) if panel_csv_path.exists() else "MISSING"
        ),
    })
    write_metadata(meta, out / "staging_metadata.json")
 
    logger.info(
        "Panel '%s' outputs saved to %s", result.panel_name, out,
    )
    required_outputs = [
        out / "event_sequence.csv",
        out / "subject_stages.csv",
        out / "mixture_models.pkl",
        out / "mcmc_samples.npy",
        out / "permutation_test.json",
        out / "face_validity_results.json",
        out / "staging_metadata.json",
    ]
    missing = [str(p) for p in required_outputs if not p.exists()]
    if missing:
        raise ValidationError(
            f"Expected output files not found after save: {missing}"
        )
 
 
def save_final_outputs(
    longitudinal_df: pd.DataFrame,
    summary_df: pd.DataFrame,
) -> None:
    """Save the updated longitudinal dataset and cross-panel summary."""
    config.ensure_project_dirs()
    long_path = config.PROCESSED_DIR / "longitudinal_validation_dataset.csv"
    save_csv(longitudinal_df, long_path)
    logger.info(
        "Updated longitudinal dataset saved: %s  (%d × %d)",
        long_path, *longitudinal_df.shape,
    )
    summary_path = config.STAGING_RESULTS_DIR / "staging_summary.csv"
    save_csv(summary_df, summary_path)
    logger.info("Cross-panel staging summary saved: %s", summary_path)

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="EBM staging for OASIS-3 (MRI-only and MRI+Amyloid panels)."
    )
    p.add_argument(
        "--skip-bootstrap",
        action="store_true",
        help="Skip bootstrap resampling.  PVD will be NaN.  Much faster.",
    )
    p.add_argument(
        "--fast",
        action="store_true",
        help=(
            "Development mode: N_MCMC_ITER=1 000, N_BOOTSTRAP=10, N_PERM=100. "
            "Results are for debugging only, not publication."
        ),
    )
    return p.parse_args()
 
 
def main() -> None:
    """Top-level orchestration."""
    args = _parse_args()
 
    config.ensure_project_dirs()
 
    logger.info("=" * 70)
    logger.info(
        "03_ebm_staging.py  |  version %s%s",
        config.PROJECT_VERSION,
        "  [FAST/DEV MODE]" if args.fast else "",
    )
    logger.info("=" * 70)
    n_mcmc_iter  = 1_000    if args.fast else config.N_MCMC_ITER
    n_bootstrap  = 10       if args.fast else config.N_BOOTSTRAP
    n_permutation = 100     if args.fast else config.N_PERMUTATION
 
    logger.info(
        "Settings: n_mcmc_iter=%d  n_bootstrap=%d  n_permutation=%d  "
        "skip_bootstrap=%s",
        n_mcmc_iter, n_bootstrap, n_permutation, args.skip_bootstrap,
    )
 
    exclusion_log: list[dict[str, Any]] = []
    panels = [
        PanelConfig(
            panel_name       = PANEL_MRI,
            csv_path         = config.PROCESSED_DIR / "ebm_mri_dataset.csv",
            biomarker_z_cols = MRI_Z_COLS,
        ),
        PanelConfig(
            panel_name       = PANEL_AMYLOID,
            csv_path         = config.PROCESSED_DIR / "ebm_mri_amyloid_dataset.csv",
            biomarker_z_cols = AMYLOID_Z_COLS,
        ),
    ]
    long_path = config.PROCESSED_DIR / "longitudinal_validation_dataset.csv"
    if not long_path.exists():
        raise FileNotFoundError(
            f"longitudinal_validation_dataset.csv not found: {long_path}\n"
            "Run 02_feature_engineering.py first."
        )
    longitudinal_df = pd.read_csv(long_path, low_memory=False)
    log_dataframe_shape(logger, longitudinal_df, "longitudinal_validation_dataset")
    results: dict[str, PanelResult] = {}
    for panel_cfg in panels:
        result = run_panel(
            panel_cfg      = panel_cfg,
            n_mcmc_iter    = n_mcmc_iter,
            n_bootstrap    = n_bootstrap,
            n_permutation  = n_permutation,
            skip_bootstrap = args.skip_bootstrap,
            exclusion_log  = exclusion_log,
        )
        save_panel_outputs(result, panel_cfg.csv_path)
        results[panel_cfg.panel_name] = result
    longitudinal_df = merge_stages_to_longitudinal(
        longitudinal_df,
        mri_result     = results[PANEL_MRI],
        amyloid_result = results[PANEL_AMYLOID],
    )
 
    summary_df = build_cross_panel_summary(
        mri_result     = results[PANEL_MRI],
        amyloid_result = results[PANEL_AMYLOID],
    )
 
    save_final_outputs(longitudinal_df, summary_df)
    logger.info("=" * 70)
    logger.info("03_ebm_staging.py COMPLETE")
    for pname, res in results.items():
        order_str = " → ".join(res.biomarker_names[i] for i in res.ml_sequence)
        logger.info(
            "  [%s]  seq: %s  |  p_perm=%.4f  |  rho_CDR=%.3f  |  converged=%s",
            pname, order_str,
            res.permutation_result.get("p_value", float("nan")),
            res.face_validity.get("spearman_rho_cdr", float("nan")),
            res.converged,
        )
    logger.info("=" * 70)
 
 
if __name__ == "__main__":
    main()

 
























