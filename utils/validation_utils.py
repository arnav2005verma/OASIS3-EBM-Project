from __future__ import annotations
import pandas as pd
class ValidationError(Exception):
    """Raised when a validation assertion fails.

    Provides a human-readable message describing what was expected and what
    was actually found, so callers can log or surface the problem without
    additional formatting.
    """


def assert_columns_present(
    df: pd.DataFrame,
    required_columns: list[str],
) -> None:
    """Assert that every column in *required_columns* exists in *df*.

    Args:
        df:               DataFrame to inspect.
        required_columns: Column names that must all be present.

    Raises:
        ValidationError: If one or more required columns are absent.
                         The message lists every missing column.
    """
    missing = [col for col in required_columns if col not in df.columns]
    if missing:
        raise ValidationError(
            f"Required column(s) missing from DataFrame: {missing}. "
            f"Columns present: {sorted(df.columns.tolist())}"
        )


def assert_no_duplicate_id(df: pd.DataFrame, id_col: str) -> None:
    """Assert that *id_col* contains no duplicate values in *df*.

    Args:
        df:     DataFrame to inspect.
        id_col: Name of the identifier column.

    Raises:
        ValidationError: If duplicates are found.  The message includes the
                         count of duplicated values and up to ten examples.
    """
    if id_col not in df.columns:
        raise ValidationError(
            f"Identifier column '{id_col}' is not present in the DataFrame. "
            f"Columns present: {sorted(df.columns.tolist())}"
        )

    duplicated_mask = df[id_col].duplicated(keep=False)
    if duplicated_mask.any():
        examples = sorted(df.loc[duplicated_mask, id_col].unique().tolist())
        raise ValidationError(
            f"Column '{id_col}' contains {duplicated_mask.sum()} duplicate "
            f"row(s) across {len(examples)} distinct value(s). "
            f"Examples (up to 10): {examples[:10]}"
        )


def assert_value_within_range(
    series: pd.Series,
    min_value: float,
    max_value: float,
) -> None:
    """Assert that every non-null value in *series* lies within [min_value, max_value].

    Null values are ignored; only non-null entries are evaluated.

    Args:
        series:    Series whose values to check.
        min_value: Minimum permitted value (inclusive).
        max_value: Maximum permitted value (inclusive).

    Raises:
        ValidationError: If any non-null value falls outside the permitted
                         range.  The message includes the count of violations
                         and the observed minimum and maximum.
    """
    non_null = series.dropna()
    if non_null.empty:
        return

    out_of_range = non_null[(non_null < min_value) | (non_null > max_value)]
    if not out_of_range.empty:
        raise ValidationError(
            f"Series '{series.name}' contains {len(out_of_range)} value(s) "
            f"outside [{min_value}, {max_value}]. "
            f"Observed range: [{non_null.min()}, {non_null.max()}]. "
            f"Example offending values: {out_of_range.head(5).tolist()}"
        )


def assert_non_empty_dataframe(df: pd.DataFrame, name: str) -> None:
    """Assert that *df* contains at least one row.

    Args:
        df:   DataFrame to check.
        name: Human-readable label for *df*, used in the error message.

    Raises:
        ValidationError: If *df* has zero rows.
    """
    if df.empty:
        raise ValidationError(
            f"DataFrame '{name}' is empty (0 rows). "
            "Upstream filtering may have removed all records."
        )