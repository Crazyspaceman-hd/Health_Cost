"""Tests for the visualization layer (Phase 5).

Charts are hard to assert pixel-by-pixel, so these verify the contract that
matters for a pipeline: every expected figure is produced as a non-empty PNG,
and the completeness filter that the charts rely on behaves correctly.
"""
from __future__ import annotations

import pandas as pd
import pytest

from src import config as C
from src import generate_data as G
from src import metrics as M
from src import visualize as Vz

EXPECTED = {
    "pmpm_by_year", "growth_vs_target", "payer_cost_growth",
    "provider_cost_growth", "provider_category_growth",
    "provider_category_dollar_growth",
    "service_category_growth", "utilization_per_1000", "price_vs_utilization",
    "validation_by_severity", "validation_resolution",
    "high_cost_concentration", "high_cost_by_facility",
    "payer_provider_heatmap",
}


@pytest.fixture(scope="module")
def cfg() -> dict:
    return C.load_config()


@pytest.fixture(scope="module")
def data(cfg) -> dict[str, pd.DataFrame]:
    return G.build_all(cfg)


def test_all_figures_written(tmp_path, monkeypatch, data, cfg):
    import os

    monkeypatch.setattr(C, "FIG_DIR", tmp_path)
    written = Vz.build_all_figures(data, cfg)
    assert set(written) == EXPECTED
    for path in written.values():
        assert os.path.exists(path)
        assert os.path.getsize(path) > 1000  # a real PNG, not an empty stub


def test_completeness_filter_removes_flagged_cell(data, cfg):
    """The incomplete PAY002 2023 cell is dropped from claims and enrollment."""
    cells = M.incomplete_payer_years(data, cfg)
    assert ("PAY002", 2023) in cells
    adj = M.apply_completeness_filter(data, cells)

    adj_claims_year = pd.to_datetime(
        adj["claims"]["service_date"], errors="coerce"
    ).dt.year
    pay002_2023 = (
        (adj["claims"]["payer_id"] == "PAY002") & (adj_claims_year == 2023)
    ).sum()
    assert pay002_2023 == 0
    enr = adj["enrollment"]
    assert ((enr["payer_id"] == "PAY002") & (enr["year"] == 2023)).sum() == 0
    # other payer-years are untouched
    assert ((enr["payer_id"] == "PAY002") & (enr["year"] == 2022)).sum() > 0


def test_filter_is_noop_when_no_cells(data):
    assert M.apply_completeness_filter(data, set()) is data
