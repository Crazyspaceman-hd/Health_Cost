"""SQLite loader and SQL-based analytics.

Loads the (cleaned) analytic claims and enrollment into a local SQLite database
and runs the parameterized SQL in ``src/sql`` against it. This demonstrates the
SQL side of the workflow and gives the test suite a way to cross-check the
pandas metrics against an independent SQL implementation of the same logic.
"""
from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Any

import pandas as pd

from . import config as C
from . import data_io
from . import metrics

DEFAULT_DB = C.PROCESSED_DIR / "health_cost.db"


def build_database(
    data: dict[str, pd.DataFrame] | None = None,
    cfg: dict[str, Any] | None = None,
    db_path: str | Path = DEFAULT_DB,
) -> Path:
    """Create the SQLite database from the source tables.

    Loads dimension tables, the full ``enrollment`` (member-month) table, and a
    cleaned ``analytic_claims`` table (raw claims with Phase 2 defects removed,
    plus a precomputed ``year`` column for convenient SQL grouping).
    """
    if cfg is None:
        cfg = C.load_config()
    if data is None:
        data = data_io.load_raw()

    C.ensure_dirs()
    db_path = Path(db_path)
    analytic = metrics.prepare_analytic_claims(data, cfg)
    # store dates as ISO strings so SQLite handles them predictably
    analytic = analytic.copy()
    analytic["service_date"] = analytic["service_date"].dt.strftime("%Y-%m-%d")

    with sqlite3.connect(db_path) as conn:
        data["payer"].to_sql("payer", conn, if_exists="replace", index=False)
        data["provider_org"].to_sql(
            "provider_org", conn, if_exists="replace", index=False
        )
        data["service_category"].to_sql(
            "service_category", conn, if_exists="replace", index=False
        )
        data["enrollment"].to_sql(
            "enrollment", conn, if_exists="replace", index=False
        )
        analytic.to_sql("analytic_claims", conn, if_exists="replace", index=False)
        # indexes that matter for the grouped aggregations
        conn.execute("CREATE INDEX IF NOT EXISTS ix_claims_payer ON analytic_claims(payer_id, year)")
        conn.execute("CREATE INDEX IF NOT EXISTS ix_enr_payer ON enrollment(payer_id, year)")
    return db_path


def run_sql_file(
    filename: str,
    params: dict[str, Any] | None = None,
    db_path: str | Path = DEFAULT_DB,
) -> pd.DataFrame:
    """Execute a named .sql file from src/sql against the database."""
    sql = (C.SQL_DIR / filename).read_text(encoding="utf-8")
    with sqlite3.connect(db_path) as conn:
        return pd.read_sql_query(sql, conn, params=params or {})


def main() -> None:
    cfg = C.load_config()
    db = build_database(cfg=cfg)
    trunc = {"trunc": cfg["truncation_threshold"]}
    print(f"SQLite database built -> {db}\n")
    print("PMPM by payer (SQL, truncated):")
    df = run_sql_file("pmpm_by_payer.sql", params=trunc, db_path=db)
    with pd.option_context("display.width", 140):
        print(df.round(2).to_string(index=False))


if __name__ == "__main__":
    main()
