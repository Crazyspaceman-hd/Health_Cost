"""Tests for the data-quality validation engine (Phase 2).

Two complementary fixtures:
  * ``defective`` — the standard generated dataset, which has injected defects.
    Every "FAIL" check should find at least its seeded issues.
  * ``clean`` — the same generator with all defect rates zeroed. The injected-
    defect checks should then report zero, proving the engine isn't flagging
    structurally valid data (no false positives on the clean core).
"""
from __future__ import annotations

import copy

import pandas as pd
import pytest

from src import config as C
from src import generate_data as G
from src import validate as V


@pytest.fixture(scope="module")
def cfg() -> dict:
    return C.load_config()


@pytest.fixture(scope="module")
def defective(cfg) -> dict[str, pd.DataFrame]:
    return G.build_all(cfg)


@pytest.fixture(scope="module")
def clean_results(cfg):
    """Validation results on a dataset generated with NO injected defects."""
    clean_cfg = copy.deepcopy(cfg)
    for key in (
        "duplicate_claim_rate", "negative_amount_rate", "paid_gt_allowed_rate",
        "missing_service_cat_rate", "orphan_claim_rate", "missing_provider_rate",
        "invalid_date_rate", "enrollment_gap_fraction",
    ):
        clean_cfg["defects"][key] = 0.0
    clean_cfg["defects"]["incomplete_submission"]["keep_fraction"] = 1.0
    data = G.build_all(clean_cfg)
    return {r.check_id: r for r in V.run_validation(data, clean_cfg)}


@pytest.fixture(scope="module")
def results(defective, cfg):
    return {r.check_id: r for r in V.run_validation(defective, cfg)}


# ---------------------------------------------------------------------------
# Engine structure
# ---------------------------------------------------------------------------
def test_all_checks_run(results):
    # one result per registered check, unique ids
    assert len(results) == len(V.CHECKS)
    assert len(set(results)) == len(V.CHECKS)


def test_severities_are_valid(results):
    for r in results.values():
        assert r.severity in V.SEVERITY_RANK


def test_status_logic(results):
    for r in results.values():
        if r.issue_count == 0:
            assert r.status == "PASS"
        elif r.severity in ("LOW", "INFO"):
            assert r.status == "REVIEW"
        else:
            assert r.status == "FAIL"


# ---------------------------------------------------------------------------
# Each defect check finds its seeded issues, and counts are exact where we
# can recompute them independently.
# ---------------------------------------------------------------------------
def test_duplicates_detected(defective, results):
    c = defective["claims"]
    expected = int(c["claim_id"].duplicated(keep="first").sum())
    assert results["DQ001"].issue_count == expected > 0


def test_negative_amounts_detected(defective, results):
    c = defective["claims"]
    expected = int(((c["allowed_amount"] < 0) | (c["paid_amount"] < 0)).sum())
    assert results["DQ002"].issue_count == expected > 0


def test_paid_gt_allowed_detected(results):
    assert results["DQ003"].issue_count > 0


def test_missing_provider_detected(defective, results):
    c = defective["claims"]
    assert results["DQ005"].issue_count == int(c["provider_org_id"].isna().sum()) > 0


def test_missing_service_category_detected(defective, results):
    c = defective["claims"]
    assert results["DQ006"].issue_count == int(c["service_category"].isna().sum()) > 0


def test_invalid_dates_detected(results):
    assert results["DQ007"].issue_count > 0


def test_orphans_detected(defective, results):
    c = defective["claims"]
    enrolled = set(defective["enrollment"]["member_id"])
    expected = int((~c["member_id"].isin(enrolled)).sum())
    assert results["DQ008"].issue_count == expected > 0


def test_enrollment_gaps_detected(results):
    assert results["DQ010"].issue_count > 0


def test_high_cost_outliers_detected(defective, cfg, results):
    c = defective["claims"]
    expected = int((c["allowed_amount"] > cfg["truncation_threshold"]).sum())
    assert results["DQ011"].issue_count == expected > 0
    assert results["DQ011"].status == "REVIEW"  # informational, not a hard fail


def test_yoy_volume_change_flags_incomplete_payer(defective, cfg, results):
    """The seeded incomplete submission (PAY002, 2023) must be flagged."""
    r = results["DQ013"]
    assert r.issue_count >= 1
    inc = cfg["defects"]["incomplete_submission"]
    expected_id = f"{inc['payer_id']}|{inc['year']}"
    assert expected_id in set(r.details["entity_id"])


def test_missing_payer_passes(results):
    # the generator never drops payer_id, so this check should be clean
    assert results["DQ004"].issue_count == 0
    assert results["DQ004"].status == "PASS"


# ---------------------------------------------------------------------------
# No false positives on a clean dataset
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "check_id",
    ["DQ001", "DQ002", "DQ003", "DQ005", "DQ006", "DQ007", "DQ008",
     "DQ009", "DQ010", "DQ013"],
)
def test_clean_dataset_has_no_seeded_defects(clean_results, check_id):
    assert clean_results[check_id].issue_count == 0
    assert clean_results[check_id].status == "PASS"


# ---------------------------------------------------------------------------
# Report artifacts
# ---------------------------------------------------------------------------
def test_summary_schema_and_sorting(results):
    summary = V.build_summary(list(results.values()))
    required = {
        "check_id", "check_name", "dimension", "severity", "status",
        "issue_count", "denominator", "affected_pct", "description",
        "recommended_follow_up",
    }
    assert required.issubset(summary.columns)
    # sorted by severity rank (most severe first)
    ranks = summary["severity"].map(V.SEVERITY_RANK).tolist()
    assert ranks == sorted(ranks)


def test_summary_has_resolution_columns(results):
    summary = V.build_summary(list(results.values()))
    assert {"handling", "resolved_count"}.issubset(summary.columns)
    # resolved_count never exceeds the count found
    assert (summary["resolved_count"] <= summary["issue_count"]).all()
    # every CRITICAL issue is neutralized before analysis (resolved == found)
    crit = summary[summary["severity"] == "CRITICAL"]
    assert (crit["resolved_count"] == crit["issue_count"]).all()
    # duplicates are fully de-duplicated
    dq001 = summary[summary["check_id"] == "DQ001"].iloc[0]
    assert dq001["resolved_count"] == dq001["issue_count"] > 0


def test_issue_detail_schema(results):
    detail = V.build_issue_detail(list(results.values()))
    assert {"check_id", "check_name", "severity", "entity_type",
            "entity_id", "detail"}.issubset(detail.columns)
    # every detail row references a real check id
    assert set(detail["check_id"]).issubset(set(results))


def test_write_outputs_creates_files(tmp_path, monkeypatch, results):
    """write_outputs should produce both CSVs in the outputs directory."""
    monkeypatch.setattr(C, "OUTPUT_DIR", tmp_path)
    summary = V.build_summary(list(results.values()))
    detail = V.build_issue_detail(list(results.values()))
    paths = V.write_outputs(summary, detail)
    assert (tmp_path / "validation_summary.csv").exists()
    assert (tmp_path / "data_quality_issues.csv").exists()
    # round-trips back to the same row counts
    assert len(pd.read_csv(paths["summary"])) == len(summary)
