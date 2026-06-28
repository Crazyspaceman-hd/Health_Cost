"""End-to-end pipeline runner.

Runs the whole workflow in one shot, sharing a single in-memory copy of the
generated data across stages for consistency and speed:

    generate -> validate -> load SQLite -> metrics -> visualize -> report

Run with ``python -m src.pipeline`` (or ``make all`` plus ``make report``).
"""
from __future__ import annotations

from typing import Any

from . import config as C
from . import generate_data as G
from . import validate as V
from . import load_db
from . import metrics as M
from . import visualize as Vz
from . import report as R


def run_pipeline(cfg: dict[str, Any] | None = None) -> dict[str, Any]:
    """Execute every stage and return a small dict of result paths/counts."""
    if cfg is None:
        cfg = C.load_config()
    C.ensure_dirs()
    out: dict[str, Any] = {}

    print("[1/6] Generating synthetic source tables...")
    data = G.build_all(cfg)
    G.write_raw(data)
    out["claims"] = len(data["claims"])
    out["member_months"] = len(data["enrollment"])

    print("[2/6] Running data-quality validation...")
    results = V.run_validation(data, cfg)
    V.write_outputs(V.build_summary(results), V.build_issue_detail(results))
    out["checks_failed"] = int(sum(r.status == "FAIL" for r in results))

    print("[3/6] Building SQLite analytics database...")
    out["db"] = str(load_db.build_database(
        data, cfg, db_path=C.PROCESSED_DIR / "health_cost.db"))

    print("[4/6] Computing cost & utilization metrics...")
    # raw tables, plus completeness-adjusted tables for reporting/charts
    M.write_metric_tables(M.build_metric_tables(data, cfg))
    out["tables"] = list(M.build_metric_tables(data, cfg, exclude_incomplete=True))

    print("[5/6] Rendering figures...")
    out["figures"] = list(Vz.build_all_figures(data, cfg))

    print("[6/6] Writing executive summary...")
    out["report"] = R.write_report(R.build_report(data, cfg))

    print("\nPipeline complete.")
    print(f"  claims={out['claims']:,}  member-months={out['member_months']:,}  "
          f"checks failed={out['checks_failed']}")
    print(f"  metric tables: {len(out['tables'])}   figures: {len(out['figures'])}")
    print(f"  database -> {out['db']}")
    print(f"  report   -> {out['report']}")
    return out


def main() -> None:
    run_pipeline()


if __name__ == "__main__":
    main()
