"""Unit tests for scripts/01_data_prep.py.

These cover the two pieces of pipeline logic that are most likely to fail
silently: diagnosis derivation from raw clinical flags, and the biomarker
z-score sign convention that every downstream EBM result in
scripts/03_ebm_staging.py depends on. A bug in either would not raise an
exception; it would just quietly produce a wrong but plausible-looking
dataset, so these are exercised directly rather than only checked at the
"does the pipeline run" level.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import config  # noqa: E402


def _load_data_prep_module():
    """Import scripts/01_data_prep.py under a valid module name.

    The filename starts with a digit, so it cannot be imported with a plain
    `import` statement or `python -m`.
    """
    path = PROJECT_ROOT / "scripts" / "01_data_prep.py"
    spec = importlib.util.spec_from_file_location("data_prep", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


data_prep = _load_data_prep_module()

_ALL_FLAG_COLS = (
    [config.NORMCOG_COL, config.DEMENTED_COL, config.PROBAD_COL, config.POSSAD_COL]
    + config.MCI_FLAG_COLS
    + config.NON_AD_EXCLUSION_COLS
)


def _flag_row(**overrides) -> dict:
    """One diagnosis-flag row with every flag defaulted to 0 (absent)."""
    row = {col: 0 for col in _ALL_FLAG_COLS}
    row.update(overrides)
    return row


class TestDeriveDiagnosisGroups:
    def test_normal_cognition_and_not_demented_is_cn(self):
        df = pd.DataFrame([_flag_row(NORMCOG=1, DEMENTED=0)])
        result = data_prep.derive_diagnosis_groups(df)
        assert result["diagnosis_group"].tolist() == [config.DX_CN]

    def test_impaired_not_demented_with_mci_flag_is_cind(self):
        df = pd.DataFrame([_flag_row(NORMCOG=0, DEMENTED=0, MCIAMEM=1)])
        result = data_prep.derive_diagnosis_groups(df)
        assert result["diagnosis_group"].tolist() == [config.DX_CIND]

    def test_demented_with_probad_is_ad(self):
        df = pd.DataFrame([_flag_row(NORMCOG=0, DEMENTED=1, PROBAD=1)])
        result = data_prep.derive_diagnosis_groups(df)
        assert result["diagnosis_group"].tolist() == [config.DX_AD]

    def test_demented_without_ad_or_exclusion_flags_is_other(self):
        df = pd.DataFrame([_flag_row(NORMCOG=0, DEMENTED=1)])
        result = data_prep.derive_diagnosis_groups(df)
        assert result["diagnosis_group"].tolist() == [config.DX_OTHER]

    def test_non_ad_exclusion_flag_overrides_ad_flag(self):
        # A subject flagged both PROBAD (probable Alzheimer's) and VASC
        # (vascular dementia) must resolve to OTHER_Dementia, not AD. The
        # non-AD etiology flag is applied last in derive_diagnosis_groups
        # and is meant to override a comorbid AD flag, not lose to it.
        df = pd.DataFrame([_flag_row(NORMCOG=0, DEMENTED=1, PROBAD=1, VASC=1)])
        result = data_prep.derive_diagnosis_groups(df)
        assert result["diagnosis_group"].tolist() == [config.DX_OTHER]

    def test_no_matching_flags_is_unclassified(self):
        df = pd.DataFrame([_flag_row(NORMCOG=0, DEMENTED=0)])
        result = data_prep.derive_diagnosis_groups(df)
        assert result["diagnosis_group"].tolist() == [config.DX_UNCLASSIFIED]

    def test_missing_flag_values_are_treated_as_absent(self):
        row = _flag_row(NORMCOG=1, DEMENTED=0)
        row["PROBAD"] = np.nan
        df = pd.DataFrame([row])
        result = data_prep.derive_diagnosis_groups(df)
        assert result["diagnosis_group"].tolist() == [config.DX_CN]


class TestBuildCnReferenceMask:
    @staticmethod
    def _rows(n: int, *, normal: bool) -> pd.DataFrame:
        if normal:
            return pd.DataFrame({
                config.NORMCOG_COL: [1] * n,
                config.DEMENTED_COL: [0] * n,
                config.CDR_GLOBAL_COL: [0.0] * n,
                config.MMSE_COL: [30.0] * n,
            })
        return pd.DataFrame({
            config.NORMCOG_COL: [0] * n,
            config.DEMENTED_COL: [1] * n,
            config.CDR_GLOBAL_COL: [1.0] * n,
            config.MMSE_COL: [20.0] * n,
        })

    def test_selects_only_strictly_normal_subjects(self):
        df = pd.concat(
            [self._rows(config.MIN_CN_REFERENCE_N, normal=True), self._rows(5, normal=False)],
            ignore_index=True,
        )
        mask = data_prep.build_cn_reference_mask(df)
        assert mask.sum() == config.MIN_CN_REFERENCE_N
        assert mask.iloc[: config.MIN_CN_REFERENCE_N].all()
        assert not mask.iloc[config.MIN_CN_REFERENCE_N:].any()

    def test_raises_when_reference_group_too_small(self):
        df = self._rows(config.MIN_CN_REFERENCE_N - 1, normal=True)
        with pytest.raises(data_prep.ValidationError):
            data_prep.build_cn_reference_mask(df)


# One generation rule per biomarker: intercept, linear age effect, ICV
# effect, and noise scale, loosely modeled on real OASIS-3 magnitudes.
# ventricular_vol is the only higher-is-abnormal, skewed biomarker; the
# other five are lower-is-abnormal.
_BIOMARKER_PARAMS = {
    "hippocampus_vol": dict(intercept=4200, age_coef=-6, icv_coef=0.0020, noise_sd=15),
    "entorhinal_thickness": dict(intercept=3.6, age_coef=-0.01, icv_coef=0.0, noise_sd=0.05),
    "fusiform_vol": dict(intercept=17_000, age_coef=-40, icv_coef=0.0030, noise_sd=200),
    "inferiortemporal_vol": dict(intercept=15_000, age_coef=-35, icv_coef=0.0030, noise_sd=200),
    "whole_brain_vol": dict(intercept=950_000, age_coef=-1500, icv_coef=0.35, noise_sd=8_000),
    "ventricular_vol": dict(intercept=8_000, age_coef=40, icv_coef=0.0010, noise_sd=300),
}


def _synthetic_cohort(n_cn: int = 40) -> pd.DataFrame:
    """Build a synthetic baseline dataframe with all six MRI biomarkers.

    n_cn cognitively normal subjects, whose biomarkers are a linear function
    of age and ICV plus noise, and one clearly abnormal (AD-like) subject
    whose biomarkers are shifted about ten noise-SDs in the pathological
    direction for every measure: atrophied for the five lower-is-abnormal
    volumes/thickness, enlarged for ventricular_vol.
    """
    rng = np.random.default_rng(42)
    age = np.linspace(60, 85, n_cn)
    icv = np.linspace(1_400_000, 1_600_000, n_cn)
    sex = np.where(np.arange(n_cn) % 2 == 0, "M", "F")

    cn_data = {
        config.ID_COL: [f"CN{i}" for i in range(n_cn)],
        config.NORMCOG_COL: 1,
        config.DEMENTED_COL: 0,
        config.CDR_GLOBAL_COL: 0.0,
        config.MMSE_COL: 30.0,
        "age_at_baseline": age,
        "sex": sex,
        "icv": icv,
    }
    abnormal_age = 75.0
    abnormal_icv = 1_500_000.0
    abnormal_data = {
        config.ID_COL: ["AD0"],
        config.NORMCOG_COL: 0,
        config.DEMENTED_COL: 1,
        config.CDR_GLOBAL_COL: 1.0,
        config.MMSE_COL: 20.0,
        "age_at_baseline": [abnormal_age],
        "sex": ["F"],
        "icv": [abnormal_icv],
    }

    for biomarker, p in _BIOMARKER_PARAMS.items():
        noise = rng.normal(0, p["noise_sd"], n_cn)
        cn_data[biomarker] = (
            p["intercept"] + p["age_coef"] * age + p["icv_coef"] * icv + noise
        )
        predicted_abnormal = (
            p["intercept"] + p["age_coef"] * abnormal_age + p["icv_coef"] * abnormal_icv
        )
        shift = 10 * p["noise_sd"]
        if biomarker in data_prep._HIGHER_IS_ABNORMAL:
            abnormal_data[biomarker] = [predicted_abnormal + shift]
        else:
            abnormal_data[biomarker] = [predicted_abnormal - shift]

    return pd.concat(
        [pd.DataFrame(cn_data), pd.DataFrame(abnormal_data)], ignore_index=True
    )


@pytest.fixture(scope="module")
def staged_cohort():
    """Run the real residualization + z-scoring pipeline on synthetic data."""
    df = _synthetic_cohort()
    df = data_prep.prepare_covariates(df)
    df = data_prep.apply_log_transforms(df)
    cn_mask = data_prep.build_cn_reference_mask(df)
    df, coef_records, _ = data_prep.residualize_all_mri(df, cn_mask)
    df, _ = data_prep.zscore_mri_biomarkers(df, cn_mask, coef_records)
    abnormal_idx = df.index[df[config.ID_COL] == "AD0"][0]
    return df, cn_mask, abnormal_idx


class TestZscoreDirectionConvention:
    """After z-scoring, a higher value must always mean more Alzheimer's-like
    pathology, whether the raw biomarker is normally lower in disease
    (hippocampal volume) or higher (ventricular volume). A sign error on
    either branch would silently corrupt every downstream EBM stage without
    raising an exception, which is exactly why this is worth locking down
    with a test rather than trusting by inspection.
    """

    @pytest.mark.parametrize(
        "biomarker",
        ["hippocampus_vol", "entorhinal_thickness", "fusiform_vol", "inferiortemporal_vol", "whole_brain_vol"],
    )
    def test_atrophy_maps_to_positive_z_for_lower_is_abnormal_biomarkers(
        self, staged_cohort, biomarker
    ):
        df, _, abnormal_idx = staged_cohort
        assert df.loc[abnormal_idx, f"z_{biomarker}"] > 0

    def test_enlargement_maps_to_positive_z_for_ventricular_volume(self, staged_cohort):
        df, _, abnormal_idx = staged_cohort
        assert df.loc[abnormal_idx, "z_ventricular_vol"] > 0

    def test_cn_reference_group_is_standardized_to_mean_zero_sd_one(self, staged_cohort):
        df, cn_mask, _ = staged_cohort
        cn_z = df.loc[cn_mask, "z_hippocampus_vol"]
        assert cn_z.mean() == pytest.approx(0.0, abs=1e-8)
        assert cn_z.std(ddof=1) == pytest.approx(1.0, abs=1e-8)
