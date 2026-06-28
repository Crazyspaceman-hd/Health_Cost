"""Configuration loading and shared path constants.

All tunable parameters live in ``config.yaml`` at the repo root. This module
loads that file once and exposes the project's directory layout as importable
constants so every other module agrees on where data and outputs go.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

# ---------------------------------------------------------------------------
# Directory layout. PROJECT_ROOT is the repo root (one level above /src).
# ---------------------------------------------------------------------------
PROJECT_ROOT: Path = Path(__file__).resolve().parents[1]

CONFIG_PATH: Path = PROJECT_ROOT / "config.yaml"
DATA_DIR: Path = PROJECT_ROOT / "data"
RAW_DIR: Path = DATA_DIR / "raw"
PROCESSED_DIR: Path = DATA_DIR / "processed"
OUTPUT_DIR: Path = PROJECT_ROOT / "outputs"
FIG_DIR: Path = OUTPUT_DIR / "figures"
REPORT_DIR: Path = PROJECT_ROOT / "reports"
SQL_DIR: Path = PROJECT_ROOT / "src" / "sql"

# File names for the six synthetic source tables written to data/raw.
RAW_FILES: dict[str, str] = {
    "payer": "dim_payer.csv",
    "provider_org": "dim_provider_org.csv",
    "service_category": "ref_service_category.csv",
    "enrollment": "fact_enrollment.csv",
    "claims": "fact_claims.csv",
    "condition": "ref_condition_group.csv",
}


def load_config(path: str | Path | None = None) -> dict[str, Any]:
    """Load and return the YAML configuration as a plain dict.

    Parameters
    ----------
    path:
        Optional override path. Defaults to ``config.yaml`` at the repo root.
    """
    cfg_path = Path(path) if path is not None else CONFIG_PATH
    if not cfg_path.exists():
        raise FileNotFoundError(f"Config file not found: {cfg_path}")
    with cfg_path.open("r", encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)
    if not isinstance(cfg, dict):
        raise ValueError(f"Config did not parse to a mapping: {cfg_path}")
    return cfg


def ensure_dirs() -> None:
    """Create the standard output directories if they do not yet exist."""
    for d in (RAW_DIR, PROCESSED_DIR, OUTPUT_DIR, FIG_DIR, REPORT_DIR):
        d.mkdir(parents=True, exist_ok=True)
