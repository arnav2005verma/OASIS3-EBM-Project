from __future__ import annotations
import json
import platform 
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
import pandas as pd
def ensure_directory(path: Path) -> None:
    """Create *path* and any missing parents. No-op if *path* already exists.
    Args:
        path: Directory to create.
    """
    path.mkdir(parents=True, exist_ok=True)
def save_csv(df: pd.DataFrame, path: Path, index: bool = False) -> None:
    """Write *df* to *path* as a CSV file.
    Parent directories are created automatically.
    Args:
        df: DataFrame to serialise.
        path: Destination file path.
        index: Whether to include the DataFrame index. Defaults to "False".
    """
    ensure_directory(path.parent)
    df.to_csv(path, index=index)
def save_json(obj: dict[str, Any], path: Path) -> None:
    """Serialise *obj* to a pretty-printed JSON file at *path*
    Parent directories are created automatically.
    Args:
        obj: Dictionary to serialise. Values must be JSON-compatible. 
        path: Destination file path. 
    """
    ensure_directory(path.parent)
    with path.open("w", encoding="utf-8") as fh:
        json.dump(obj, fh, indent=2, default=str)
def load_json(path: Path) -> dict[str, Any]:
    """Load and return the JSON file at *path*.
    Args:
        path: Path to an existing JSON file.
    Returns:
           Parsed content as a dictionary. 
    Raises: 
          FileNotFoundError: If *path* does not exist.
          json.JSONDecodeError: If the file is not valid JSON.
    """
    if not path.exists():
        raise FileNotFoundError(f"JSON file not found: {path}")
    with path.open("r", encoding="utf-8") as fh:
        return json.load(fh)
def write_metadata(metadata: dict[str, Any], path: Path) -> None:
    """Write a metadata dictionary to a JSON file at *path*.
    Thin wrapper around :func: 'save_json' provided for naming clarity at call sites that produce provenance records.
    Args:
        metadata: Metadata dictionary to persist.
        path: Destination file path.
    """
    save_json(metadata, path)
def build_run_metadata(script_name: str) -> dict[str, Any]:
    """Construct a provenance metadata record for a pipeline run.
    Args: 
        script_name: Identifier for the script producing this record, e.g. ''"00_extract_merge.py"''.
    Returns: 
       Dictionary containing: 
       * ``script_name``   - the supplied identifier.
        * ``timestamp_utc`` - ISO-8601 timestamp at call time (UTC).
        * ``python_version`` - full Python version string.
        * ``platform``      - operating system description.
    """
    return {
        "script_name": script_name,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "python_version": sys.version,
        "platform": platform.platform(),
    }
def save_environment_snapshot(output_dir: Path) -> None:
    """Capture the current Python environment and write it to *output_dir*.
    Creates two files:

    * ``environment.json``       - interpreter version and platform details.
    * ``installed_packages.txt`` - ``pip freeze`` output, one package per line.

    If ``pip freeze`` cannot be executed the packages file contains an
    explanatory message rather than raising.

    Args:
        output_dir: Directory in which to place both files.
    """
    ensure_directory(output_dir)
    env_info: dict[str, Any] = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "python_version": sys.version,
        "platform": platform.platform(),
        "system": platform.system(),
        "release": platform.release(),
        "machine": platform.machine(),
    }
    save_json(env_info, output_dir / "environment.json")
    try:
        result = subprocess.run(
            [sys.executable, "-m", "pip", "freeze"],
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
        packages_text = result.stdout if result.returncode == 0 else (
            f"pip freeze exited with code {result.returncode}.\n"
            f"stderr: {result.stderr}"
        )
    except (FileNotFoundError, subprocess.TimeoutExpired) as exc:
        packages_text = f"Could not run pip freeze: {exc}"
    (output_dir / "installed_packages.txt").write_text(packages_text, encoding="utf-8")

