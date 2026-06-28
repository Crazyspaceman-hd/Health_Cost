"""Smoke test for the Streamlit dashboard.

Uses Streamlit's AppTest to actually execute the app script headlessly and
assert it renders without raising. Skipped if Streamlit isn't installed.
"""
from __future__ import annotations

import pytest

pytest.importorskip("streamlit")

from pathlib import Path

from streamlit.testing.v1 import AppTest

from src import config as C

APP = str(Path(C.PROJECT_ROOT) / "src" / "dashboard.py")


def test_dashboard_runs_without_exception():
    # raw tables must exist so the app loads quickly; generate if missing
    from src import data_io, generate_data as G
    try:
        data_io.load_raw()
    except FileNotFoundError:
        G.write_raw(G.build_all(C.load_config()))

    at = AppTest.from_file(APP, default_timeout=120).run()
    assert not at.exception
    # the headline title and the KPI metrics rendered
    assert any("Health Cost Growth Target Analytics" in t.value for t in at.title)
    assert len(at.metric) >= 5
    assert len(at.tabs) >= 6
