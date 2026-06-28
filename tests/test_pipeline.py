"""Tests for the executive summary generator and the end-to-end pipeline."""
from __future__ import annotations

import pandas as pd
import pytest

from src import config as C
from src import generate_data as G
from src import metrics as M
from src import report as R
from src import pipeline as P


@pytest.fixture(scope="module")
def cfg() -> dict:
    return C.load_config()


@pytest.fixture(scope="module")
def data(cfg) -> dict[str, pd.DataFrame]:
    return G.build_all(cfg)


# ---------------------------------------------------------------------------
# Report content / target-comparison narrative
# ---------------------------------------------------------------------------
def test_report_has_core_sections(data, cfg):
    text = R.build_report(data, cfg)
    for heading in (
        "Executive Summary", "Data evaluated", "Data quality", "Key findings",
        "Payers relative to target", "Service categories driving cost growth",
        "Price vs. utilization", "Recommended next analytical steps",
    ):
        assert heading in text


def test_report_classifies_payers_against_target(data, cfg):
    """Payers above target appear under 'Exceeded'; the incomplete payer is
    reported as 'Not assessed', never classified."""
    tables = M.build_metric_tables(data, cfg, exclude_incomplete=True)
    payer = tables["payer_cost_growth_summary"].drop_duplicates("payer_id")
    text = R.build_report(data, cfg)

    above = payer[payer["complete_submission"] & payer["cagr_exceeds_target"]]
    for name in above["payer_name"]:
        assert name in text

    incomplete = payer[~payer["complete_submission"]]
    if len(incomplete):
        assert "Not assessed" in text
        for name in incomplete["payer_name"]:
            assert name in text


def test_report_is_synthetic_labeled(data, cfg):
    assert "Synthetic data notice" in R.build_report(data, cfg)


def test_report_classifies_provider_growth_profile(data, cfg):
    """When providers exceed target, the report describes each one's profile
    (breadth + dominant dollar driver)."""
    tables = M.build_metric_tables(data, cfg, exclude_incomplete=True)
    provider = tables["provider_cost_growth_summary"].drop_duplicates("provider_org_id")
    text = R.build_report(data, cfg)
    if provider["cagr_exceeds_target"].any():
        assert "categories above target" in text
        assert any(k in text for k in ("broad-based", "concentrated", "mixed"))


def test_write_report(tmp_path, monkeypatch, data, cfg):
    monkeypatch.setattr(C, "REPORT_DIR", tmp_path)
    path = R.write_report(R.build_report(data, cfg))
    assert (tmp_path / "executive_summary.md").exists()
    assert len(open(path, encoding="utf-8").read()) > 500


# ---------------------------------------------------------------------------
# End-to-end pipeline produces every artifact
# ---------------------------------------------------------------------------
def test_pipeline_creates_all_outputs(tmp_path, monkeypatch, cfg):
    monkeypatch.setattr(C, "RAW_DIR", tmp_path / "raw")
    monkeypatch.setattr(C, "PROCESSED_DIR", tmp_path / "proc")
    monkeypatch.setattr(C, "OUTPUT_DIR", tmp_path / "out")
    monkeypatch.setattr(C, "FIG_DIR", tmp_path / "out" / "figures")
    monkeypatch.setattr(C, "REPORT_DIR", tmp_path / "rep")

    out = P.run_pipeline(cfg)

    # raw source tables
    for fname in C.RAW_FILES.values():
        assert (C.RAW_DIR / fname).exists()
    # validation + metric output CSVs
    for name in ("validation_summary", "data_quality_issues",
                 "payer_cost_growth_summary", "executive_summary_metrics",
                 "service_category_trends", "high_cost_concentration"):
        assert (C.OUTPUT_DIR / f"{name}.csv").exists()
    # figures, database, report
    assert len(list(C.FIG_DIR.glob("*.png"))) == 14
    assert (C.PROCESSED_DIR / "health_cost.db").exists()
    assert (C.REPORT_DIR / "executive_summary.md").exists()
    assert out["claims"] > 50_000
