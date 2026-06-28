"""Tests for the synthetic data generator (Phase 1).

These assert the structural and economic properties the rest of the pipeline
depends on: correct schemas, referential integrity, reproducibility, the
presence of the seeded data-quality defects, and that the seeded cost-growth
story (a payer above target, a payer below target) actually materializes.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src import config as C
from src import generate_data as G


@pytest.fixture(scope="module")
def cfg() -> dict:
    return C.load_config()


@pytest.fixture(scope="module")
def tables(cfg) -> dict[str, pd.DataFrame]:
    """Build the full set of tables once for the whole module."""
    return G.build_all(cfg)


# ---------------------------------------------------------------------------
# Schema / structure
# ---------------------------------------------------------------------------
def test_all_tables_present(tables):
    assert set(tables) == {
        "payer",
        "provider_org",
        "service_category",
        "condition",
        "enrollment",
        "claims",
    }


def test_dimension_counts(cfg, tables):
    assert len(tables["payer"]) == len(cfg["payers"]) >= 3
    assert len(tables["provider_org"]) == len(cfg["provider_orgs"]) >= 4
    assert len(tables["service_category"]) == len(cfg["service_categories"])


def test_claims_schema(tables):
    expected = {
        "claim_id", "member_id", "payer_id", "provider_org_id",
        "line_of_business", "region", "service_date", "enrollment_month",
        "service_category", "condition_group", "allowed_amount",
        "paid_amount", "member_cost_share", "claim_status", "received_date",
    }
    assert expected.issubset(set(tables["claims"].columns))


def test_enrollment_schema(tables):
    expected = {
        "member_id", "payer_id", "provider_org_id", "enrollment_month",
        "year", "age_band", "region", "line_of_business",
    }
    assert expected.issubset(set(tables["enrollment"].columns))
    # the internal helper column must not leak into the public table
    assert "month_idx" not in tables["enrollment"].columns


def test_meaningful_volume(tables):
    """Not a toy dataset: plenty of member-months and claims."""
    assert len(tables["enrollment"]) > 100_000
    assert len(tables["claims"]) > 50_000
    assert tables["enrollment"]["member_id"].nunique() > 5_000


def test_multiple_years_present(cfg, tables):
    years = pd.to_datetime(tables["claims"]["service_date"]).dt.year
    # every measurement year is represented
    for y in cfg["years"]:
        assert (years == y).any()


# ---------------------------------------------------------------------------
# Referential integrity (clean core, before counting injected orphans)
# ---------------------------------------------------------------------------
def test_member_ids_mostly_in_enrollment(tables):
    """All claims map to an enrolled member except the seeded orphans."""
    claim_members = set(tables["claims"]["member_id"])
    enrolled = set(tables["enrollment"]["member_id"])
    orphans = claim_members - enrolled
    # orphans exist (by design) but are a tiny minority
    assert len(orphans) > 0
    assert len(orphans) < 0.05 * len(claim_members)
    # every orphan id follows the seeded MBR9 pattern
    assert all(o.startswith("MBR9") for o in orphans)


def test_payer_provider_ids_valid(tables):
    valid_payers = set(tables["payer"]["payer_id"])
    assert set(tables["claims"]["payer_id"]).issubset(valid_payers)


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------
def test_reproducible_under_seed(cfg):
    a = G.build_all(cfg)["claims"]
    b = G.build_all(cfg)["claims"]
    assert a.shape == b.shape
    pd.testing.assert_frame_equal(a, b)


def test_different_seed_changes_data(cfg):
    cfg2 = dict(cfg)
    cfg2["random_seed"] = int(cfg["random_seed"]) + 1
    a = G.build_all(cfg)["claims"]
    b = G.build_all(cfg2)["claims"]
    assert a.shape != b.shape or not a.equals(b)


# ---------------------------------------------------------------------------
# Injected data-quality defects are actually present
# ---------------------------------------------------------------------------
def test_duplicate_claim_ids_present(tables):
    cid = tables["claims"]["claim_id"]
    assert cid.duplicated().any()


def test_negative_amounts_present(tables):
    c = tables["claims"]
    assert (c["allowed_amount"] < 0).any()


def test_paid_gt_allowed_present(tables):
    c = tables["claims"]
    # restrict to non-negative rows so we measure the seeded paid>allowed cases
    pos = c[c["allowed_amount"] > 0]
    assert (pos["paid_amount"] > pos["allowed_amount"]).any()


def test_missing_service_category_present(tables):
    assert tables["claims"]["service_category"].isna().any()


def test_missing_provider_present(tables):
    assert tables["claims"]["provider_org_id"].isna().any()


def test_invalid_dates_present(cfg, tables):
    yrs = pd.to_datetime(tables["claims"]["service_date"]).dt.year
    assert (yrs < int(cfg["years"][0])).any()


def test_enrollment_gaps_present(tables):
    """At least some members have a non-contiguous run of enrollment months."""
    enr = tables["enrollment"].copy()
    enr["period"] = pd.PeriodIndex(enr["enrollment_month"], freq="M")
    gap_members = 0
    for _, grp in enr.groupby("member_id"):
        periods = grp["period"].sort_values()
        span = (periods.max() - periods.min()).n + 1
        if span > len(periods):  # fewer months than the span implies a gap
            gap_members += 1
            if gap_members >= 1:
                break
    assert gap_members >= 1


def test_incomplete_submission_volume_drop(cfg, tables):
    """The seeded payer/year shows a large volume drop vs its prior year."""
    inc = cfg["defects"]["incomplete_submission"]
    c = tables["claims"].copy()
    c["yr"] = pd.to_datetime(c["service_date"]).dt.year
    seg = c[c["payer_id"] == inc["payer_id"]]
    vol_prior = (seg["yr"] == int(inc["year"]) - 1).sum()
    vol_target = (seg["yr"] == int(inc["year"])).sum()
    # expect a meaningful drop given keep_fraction well below 1.0
    assert vol_target < vol_prior


# ---------------------------------------------------------------------------
# The seeded cost-growth story materializes
# ---------------------------------------------------------------------------
def _payer_pmpm_by_year(
    claims: pd.DataFrame, enroll: pd.DataFrame, trunc: float = 50_000.0
) -> pd.DataFrame:
    """Quick PMPM-by-payer-year helper using clean, high-cost-truncated rows.

    Mirrors standard cost-growth methodology: catastrophic high-cost claims are
    truncated before measuring the underlying trend, because a handful of very
    large claims make single-segment year-over-year PMPM extremely volatile.
    Phase 3 metrics report both raw and truncated PMPM.
    """
    c = claims.copy()
    c = c[(c["allowed_amount"] > 0) & (c["allowed_amount"] <= trunc)]
    c["yr"] = pd.to_datetime(c["service_date"]).dt.year
    spend = c.groupby(["payer_id", "yr"])["allowed_amount"].sum()

    e = enroll.copy()
    e["yr"] = e["year"]
    mm = e.groupby(["payer_id", "yr"]).size().rename("member_months")

    df = pd.concat([spend.rename("allowed"), mm], axis=1).dropna()
    df["pmpm"] = df["allowed"] / df["member_months"]
    return df.reset_index()


def _payer_cagr(cfg, claims, enroll, payer_id) -> float:
    """Annualized truncated PMPM growth (CAGR) for one payer across all years."""
    df = _payer_pmpm_by_year(claims, enroll, trunc=cfg["truncation_threshold"])
    years = sorted(cfg["years"])
    p = df[df["payer_id"] == payer_id].set_index("yr")["pmpm"]
    return (p[years[-1]] / p[years[0]]) ** (1 / (len(years) - 1)) - 1


def test_high_trend_payer_exceeds_target(cfg, tables):
    """PAY001 is seeded clearly above the growth target."""
    cagr = _payer_cagr(cfg, tables["claims"], tables["enrollment"], "PAY001")
    assert cagr > cfg["target_growth_rate"]


def test_low_trend_payer_below_target(cfg, tables):
    """PAY003 is seeded clearly below the growth target (well-managed plan).

    PAY003 (not PAY002) is the clean low-trend signal: PAY002's 2023 submission
    is deliberately incomplete, so its trend is intentionally unreliable and is
    asserted only via the volume-drop check, not a trend check.
    """
    cagr = _payer_cagr(cfg, tables["claims"], tables["enrollment"], "PAY003")
    assert cagr < cfg["target_growth_rate"]


def test_trend_ordering_reflects_seed(cfg, tables):
    """The generator's economic logic: a payer seeded with a higher trend must
    actually grow faster than one seeded lower. This monotonicity is robust to
    sampling noise in a way an absolute threshold is not."""
    high = _payer_cagr(cfg, tables["claims"], tables["enrollment"], "PAY001")
    low = _payer_cagr(cfg, tables["claims"], tables["enrollment"], "PAY003")
    assert high > low
    # and the program has at least one payer on each side of the target
    above = high > cfg["target_growth_rate"]
    below = low < cfg["target_growth_rate"]
    assert above and below


def test_specialty_facility_high_cost_concentration(cfg, tables):
    """The configured specialty org carries a disproportionate share of
    high-cost members (a synthesized tertiary/referral case-mix pattern)."""
    org_id = cfg["high_cost"].get("specialty_org_id")
    share = float(cfg["high_cost"].get("specialty_share", 0) or 0)
    if not org_id or share <= 0:
        pytest.skip("specialty facility concentration disabled")
    c = tables["claims"]
    enr = tables["enrollment"]
    trunc = cfg["truncation_threshold"]
    member_org = enr.groupby("member_id")["provider_org_id"].first()
    spend = c[c["allowed_amount"] > 0].groupby("member_id")["allowed_amount"].sum()
    hc_members = spend[spend > trunc].index
    hc_orgs = member_org.reindex(hc_members).dropna()
    # the specialty org holds far more than a proportional 1/n_orgs share
    specialty_frac = (hc_orgs == org_id).mean()
    n_orgs = enr["provider_org_id"].nunique()
    assert specialty_frac > 2.0 / n_orgs  # at least double the even split


def test_high_cost_member_concentration(cfg, tables):
    """A small share of members should account for a disproportionate share of
    total allowed spend (supports high-cost-claimant concentration analysis)."""
    c = tables["claims"]
    c = c[c["allowed_amount"] > 0]
    by_member = c.groupby("member_id")["allowed_amount"].sum().sort_values(
        ascending=False
    )
    n_top = max(1, int(0.05 * len(by_member)))  # top 5% of members
    top_share = by_member.head(n_top).sum() / by_member.sum()
    # top 5% of members drive well more than 5% of spend (real-world skew)
    assert top_share > 0.20
