from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import pandas as pd

_LOG_FORMAT  = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"


def get_logger(
    name: str,
    log_file: Path | None = None,
    level: int = logging.INFO,
) -> logging.Logger:
    """Return a named logger with a console handler and optional file handler.

    Handlers are attached only once per logger name.  Subsequent calls with
    the same *name* return the existing logger unchanged, preventing duplicate
    log lines when modules are imported more than once.

    Args:
        name:     Logger name, conventionally ``__name__`` of the caller.
        log_file: Optional path for a log file.  The file's parent directory
                  is created automatically.  ``None`` disables file logging.
        level:    Logging level applied to the logger and all its handlers.
                  Defaults to ``logging.INFO``.

    Returns:
        Configured :class:`logging.Logger`.
    """
    logger = logging.getLogger(name)

    if logger.handlers:
        return logger

    logger.setLevel(level)
    formatter = logging.Formatter(_LOG_FORMAT, datefmt=_DATE_FORMAT)

    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    if log_file is not None:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    logger.propagate = False
    return logger


def log_dataframe_shape(
    logger: logging.Logger,
    df: pd.DataFrame,
    name: str,
) -> None:
    """Log the row and column count of a DataFrame at INFO level.

    Args:
        logger: Logger to write to.
        df:     DataFrame whose shape to report.
        name:   Human-readable label used in the log message.
    """
    logger.info("%s: %d rows x %d columns", name, df.shape[0], df.shape[1])


def log_cascading_filter(
    logger: logging.Logger,
    step_name: str,
    before_n: int,
    after_n: int,
) -> None:
    """Log the effect of a single filter step at INFO level.

    Produces a single, consistently formatted line showing how many records
    survived and how many were removed by *step_name*.

    Args:
        logger:    Logger to write to.
        step_name: Short identifier for this filter step.
        before_n:  Record count before the filter was applied.
        after_n:   Record count after the filter was applied.
    """
    removed = before_n - after_n
    pct = (removed / before_n * 100) if before_n > 0 else 0.0
    logger.info(
        "[%s] %d → %d  (removed %d, %.1f%%)",
        step_name, before_n, after_n, removed, pct,
    )


def append_exclusion_log(
    log_list: list[dict[str, Any]],
    step_name: str,
    before_n: int,
    after_n: int,
) -> None:
    """Append a filter-step record to an in-memory exclusion log.

    The caller owns *log_list* and is responsible for persisting it
    (e.g. via :func:`io_utils.save_json` or :func:`io_utils.save_csv`).

    Each appended record contains:

    * ``step``            - the *step_name* supplied by the caller.
    * ``before_n``        - record count before filtering.
    * ``after_n``         - record count after filtering.
    * ``removed_n``       - number of records removed.
    * ``percent_removed`` - fraction removed, expressed as a percentage
                            rounded to two decimal places.  ``0.0`` when
                            *before_n* is zero.

    Args:
        log_list:  Mutable list to which the record is appended in-place.
        step_name: Short identifier for the filter step.
        before_n:  Record count before the filter was applied.
        after_n:   Record count after the filter was applied.
    """
    removed_n = before_n - after_n
    percent_removed = round((removed_n / before_n * 100), 2) if before_n > 0 else 0.0
    log_list.append({
        "step":            step_name,
        "before_n":        before_n,
        "after_n":         after_n,
        "removed_n":       removed_n,
        "percent_removed": percent_removed,
    })