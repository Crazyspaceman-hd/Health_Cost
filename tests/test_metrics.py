"""Tests for the metrics engine (Phase 3).

Three layers:
  * exact arithmetic on a tiny hand-built dataset (PMPM, truncation, YoY,
    target comparison, utilization/1000 are all checkable by hand);
  * properties on the full generated dataset (clean analytic base, seeded story);
  * a SQL-vs-pandas cross-check proving both implementations agree.
"""
from __future__ import annotations

import pandas as pd
import pytest

from src import config as C
from src import generate_data as G
from src import load_db
from src import metrics as M


# ---------------------------------------------------------------------------
# Tiny, fully hand-computable dataset
# ---------------------------------------------------------------------------
@pytest.fixture
def tiny():
    """A 2-member, 2-year dataset with known PMPM arithmetic.

    Member-months: 2021 -> 3 (M1 x2, M2 x1), 2022 -> 1 (M1 x1).
    Allowed:       2021 -> M1 100, M2 300; 2022 -> M1 500.
    With truncation = 200: truncated allowed 2021 = 100+200 = 300 -> PMPM 100;
    2022 = 200 -> PMPM 200; YoY truncated growth = 200/100 - 1 = +100%.
    """
    cfg = {
        "years": [2021, 2022],
        "truncation_threshold": 200.0,
        "target_growth_rate": 0.034,
    }
    enroll = pd.DataFrame(
        {
            "member_id": ["M1", "M1", "M1", "M2"],
            "payer_id": ["P1"] * 4,
            "provider_org_id": ["O1"] * 4,
            "line_of_business": ["Commercial"] * 4,
            "region": ["R"] * 4,
            "enrollment_month": ["2021-01", "2021-02", "2022-01", "2021-01"],
            "year": [2021, 2021, 2022, 2021],
            "age_band": ["18-34"] * 4,
        }
    )
    claims = pd.DataFrame(
        {
            "claim_id": ["C1", "C2", "C3"],
            "member_id": ["M1", "M2", "M1"],
            "payer_id": ["P1", "P1", "P1"],
            "provider_org_id": ["O1", "O1", "O1"],
            "line_of_business": ["Commercial"] * 3,
            "region": ["R"] * 3,
            "service_date": ["2021-01-10", "2021-01-15", "2022-01-10"],
            "service_category": ["Inpatient", "Pharmacy", "Inpatient"],
            "allowed_amount": [100.0, 300.0, 500.0],
            "paid_amount": [90.0, 270.0, 450.0],
            "member_cost_share": [10.0, 30.0, 50.0],
            "claim_status": ["Paid"] * 3,
        }
    )
    data = {
        "claims": claims,
        "enrollment": enroll,
        "payer": pd.DataFrame({"payer_id": ["P1"], "payer_name": ["Plan"],
                               "payer_type": ["Commercial"]}),
        "provider_org": pd.DataFrame({"provider_org_id": ["O1"],
                                      "provider_org_name": ["Org"], "region": ["R"]}),
        "service_category": pd.DataFrame({"service_category": ["Inpatient", "Pharmacy"]}),
    }
    return data, cfg


def test_member_months(tiny):
    data, cfg = tiny
    mm = M.member_months(data["enrollment"], ["year"])
    assert mm.loc[2021] == 3
    assert mm.loc[2022] == 1


def test_pmpm_and_truncation(tiny):
    data, cfg = tiny
    claims = M.prepare_analytic_claims(data, cfg)
    df = M.cost_growth_by(claims, data["enrollment"], "payer_id", cfg)
    by_year = df.set_index("year")
    # raw allowed PMPM 2021 = (100+300)/3
    assert by_year.loc[2021, "allowed_pmpm"] == pytest.approx(400 / 3)
    # truncated PMPM: 2021 = (100+200)/3 = 100 ; 2022 = 200/1 = 200
    assert by_year.loc[2021, "truncated_pmpm"] == pytest.approx(100.0)
    assert by_year.loc[2022, "truncated_pmpm"] == pytest.approx(200.0)


def test_yoy_and_target_comparison(tiny):
    data, cfg = tiny
    claims = M.prepare_analytic_claims(data, cfg)
    df = M.cost_growth_by(claims, data["enrollment"], "payer_id", cfg)
    by_year = df.set_index("year")
    assert pd.isna(by_year.loc[2021, "yoy_growth"])          # no prior year
    assert by_year.loc[2022, "yoy_growth"] == pytest.approx(1.0)  # 200/100 - 1
    assert by_year.loc[2022, "exceeds_target"]               # 100% > 3.4%
    assert by_year.loc[2022, "vs_target"] == pytest.approx(1.0 - 0.034)


def test_utilization_per_1000(tiny):
    data, cfg = tiny
    claims = M.prepare_analytic_claims(data, cfg)
    trends = M.service_category_trends(claims, data["enrollment"], cfg)
    # 2021: 2 claims (Inpatient, Pharmacy) over 3 member-months -> 666.67 / 1000 MM
    util_2021 = trends[trends["year"] == 2021]["utilization_per_1000_mm"].sum()
    assert util_2021 == pytest.approx(2 / 3 * 1000)


def test_category_pmpm_sums_to_total(tiny):
    """Category truncated PMPMs should sum to the overall truncated PMPM."""
    data, cfg = tiny
    claims = M.prepare_analytic_claims(data, cfg)
    trends = M.service_category_trends(claims, data["enrollment"], cfg)
    exe = M.executive_summary_metrics(claims, data["enrollment"], cfg)
    for yr in (2021, 2022):
        cat_sum = trends[trends["year"] == yr]["truncated_pmpm"].sum()
        overall = exe.set_index("year").loc[yr, "truncated_pmpm"]
        assert cat_sum == pytest.approx(overall)


# ---------------------------------------------------------------------------
# Full generated dataset
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def cfg() -> dict:
    return C.load_config()


@pytest.fixture(scope="module")
def data(cfg) -> dict[str, pd.DataFrame]:
    return G.build_all(cfg)


def test_analytic_claims_are_clean(data, cfg):
    claims = M.prepare_analytic_claims(data, cfg)
    assert not claims["claim_id"].duplicated().any()
    assert (claims["allowed_amount"] >= 0).all()
    assert (claims["paid_amount"] >= 0).all()
    assert (claims["paid_amount"] <= claims["allowed_amount"] + 0.01).all()
    assert claims["service_category"].notna().all()
    enrolled = set(data["enrollment"]["member_id"])
    assert claims["member_id"].isin(enrolled).all()
    years = set(cfg["years"])
    assert set(claims["year"]).issubset(years)


def test_truncated_never_exceeds_allowed(data, cfg):
    tables = M.build_metric_tables(data, cfg)
    exe = tables["executive_summary_metrics"]
    assert (exe["truncated_pmpm"] <= exe["allowed_pmpm"] + 1e-6).all()


def test_high_cost_concentration_monotonic(data, cfg):
    claims = M.prepare_analytic_claims(data, cfg)
    conc = M.high_cost_concentration(claims, cfg)
    for _, row in conc.iterrows():
        assert 0 <= row["top_1pct_spend_share"] <= row["top_5pct_spend_share"]
        assert row["top_5pct_spend_share"] <= row["top_10pct_spend_share"] <= 1
    # real-world skew: top 5% of members hold well over 5% of spend
    pooled = conc[conc["period"] == "All"].iloc[0]
    assert pooled["top_5pct_spend_share"] > 0.20


def test_seeded_story_in_metrics(data, cfg):
    """PAY001 & PAY004 above target by CAGR; PAY003 below."""
    tables = M.build_metric_tables(data, cfg)
    payer = tables["payer_cost_growth_summary"].drop_duplicates("payer_id")
    flag = payer.set_index("payer_id")["cagr_exceeds_target"]
    assert flag["PAY001"] and flag["PAY004"]
    assert not flag["PAY003"]


def test_provider_category_growth(data, cfg):
    claims = M.prepare_analytic_claims(data, cfg)
    pcg = M.provider_category_growth(claims, data["enrollment"], cfg)
    assert {"provider_org_id", "service_category",
            "pmpm_start", "pmpm_end", "growth"}.issubset(pcg.columns)
    # every provider appears, paired with service categories
    assert pcg["provider_org_id"].nunique() == data["provider_org"]["provider_org_id"].nunique()
    assert pcg["growth"].notna().any()


def test_price_utilization_decomposition(data, cfg):
    claims = M.prepare_analytic_claims(data, cfg)
    dec = M.price_utilization_decomposition(claims, data["enrollment"], cfg)
    assert {"service_category", "util_growth", "price_growth", "pmpm_growth",
            "util_pts", "price_pts"}.issubset(dec.columns)
    # utilization + price contributions reconstruct PMPM growth exactly
    resid = (dec["util_pts"] + dec["price_pts"] - dec["pmpm_growth"]).abs()
    assert (resid < 1e-9).all()


def test_high_cost_by_facility(data, cfg):
    claims = M.prepare_analytic_claims(data, cfg)
    hcf = M.high_cost_by_facility(claims, data["enrollment"], cfg)
    assert {"provider_org_id", "high_cost_members", "high_cost_share",
            "concentration_index", "raw_pmpm", "truncated_pmpm"}.issubset(hcf.columns)
    # shares sum to ~1 and the specialty org is over-concentrated when enabled
    assert hcf["high_cost_share"].sum() == pytest.approx(1.0, abs=1e-6)
    assert (hcf["raw_pmpm"] >= hcf["truncated_pmpm"] - 1e-6).all()
    org_id = cfg["high_cost"].get("specialty_org_id")
    if org_id and float(cfg["high_cost"].get("specialty_share", 0) or 0) > 0:
        idx = hcf.set_index("provider_org_id").loc[org_id, "concentration_index"]
        assert idx > 1.5


def test_provider_growth_profile(data, cfg):
    claims = M.prepare_analytic_claims(data, cfg)
    prof = M.provider_growth_profile(claims, data["enrollment"], cfg)
    assert {"provider_org_id", "n_above_target", "n_categories",
            "top_category", "top_category_dollar_share"}.issubset(prof.columns)
    assert (prof["n_above_target"] <= prof["n_categories"]).all()
    # dollar share, where defined, is a valid fraction of positive growth
    s = prof["top_category_dollar_share"].dropna()
    assert ((s > 0) & (s <= 1.0 + 1e-9)).all()


def test_metric_table_set(data, cfg):
    tables = M.build_metric_tables(data, cfg)
    assert set(tables) == {
        "payer_cost_growth_summary", "provider_cost_growth_summary",
        "line_of_business_summary", "service_category_trends",
        "high_cost_concentration", "executive_summary_metrics",
    }


def test_write_metric_tables(tmp_path, monkeypatch, data, cfg):
    monkeypatch.setattr(C, "OUTPUT_DIR", tmp_path)
    tables = M.build_metric_tables(data, cfg)
    written = M.write_metric_tables(tables)
    for name in tables:
        assert (tmp_path / f"{name}.csv").exists()
    assert len(written) == len(tables)


# ---------------------------------------------------------------------------
# SQL vs pandas cross-check
# ---------------------------------------------------------------------------
def test_sql_matches_pandas_pmpm(tmp_path, data, cfg):
    """The SQL PMPM-by-payer query must agree with the pandas metrics."""
    db = load_db.build_database(data, cfg, db_path=tmp_path / "t.db")
    sql = load_db.run_sql_file(
        "pmpm_by_payer.sql", params={"trunc": cfg["truncation_threshold"]},
        db_path=db,
    ).set_index(["payer_id", "year"])

    claims = M.prepare_analytic_claims(data, cfg)
    pdf = M.cost_growth_by(claims, data["enrollment"], "payer_id", cfg).set_index(
        ["payer_id", "year"]
    )
    joined = sql.join(pdf, lsuffix="_sql", rsuffix="_pd")
    assert (joined["member_months_sql"] == joined["member_months_pd"]).all()
    pd.testing.assert_series_equal(
        joined["truncated_pmpm_sql"], joined["truncated_pmpm_pd"],
        check_names=False, rtol=1e-9,
    )
