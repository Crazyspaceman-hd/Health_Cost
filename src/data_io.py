"""Shared I/O helpers for reading the synthetic source tables.

Used by the validation engine (Phase 2) and the metrics engine (Phase 3) so
both read the raw tables the same way, with dates parsed consistently.
"""
from __future__ import annotations

import pandas as pd

from . import config as C

# Columns that must be parsed as datetimes when read from CSV.
_DATE_COLS = {
    "claims": ["service_date", "received_date"],
}


def load_raw() -> dict[str, pd.DataFrame]:
    """Load every raw source table from data/raw into a dict of DataFrames.

    Returns the same keys as ``generate_data.build_all`` so downstream code can
    accept either freshly generated in-memory tables or tables read from disk.
    """
    tables: dict[str, pd.DataFrame] = {}
    for key, fname in C.RAW_FILES.items():
        path = C.RAW_DIR / fname
        if not path.exists():
            raise FileNotFoundError(
                f"Missing raw table '{fname}'. Run `python -m src.generate_data` first."
            )
        df = pd.read_csv(path)
        for col in _DATE_COLS.get(key, []):
            if col in df.columns:
                df[col] = pd.to_datetime(df[col], errors="coerce")
        tables[key] = df
    return tables
