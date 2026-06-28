"""Synthetic healthcare cost data generator.

Produces six internally consistent source tables that resemble what a state
cost-growth program would collect from payers and provider organizations:

    dim_payer, dim_provider_org, ref_service_category, ref_condition_group,
    fact_enrollment (member-month grain), fact_claims (claim grain)

Design goals
------------
* **Reproducible** — all randomness flows from a single seeded RNG.
* **A real economic story** — payer/provider/LOB/category trends are seeded
  (see config.yaml) so the downstream analysis finds meaningful, explainable
  cost growth instead of noise. At least one payer and one provider org are
  deliberately above the program's growth target.
* **Auditable data-quality defects** — known issues (duplicates, negatives,
  orphans, an incomplete submission, etc.) are injected last so the Phase 2
  validation engine has real findings to surface.

NOTHING here is real: no PHI, no real claims, no real diagnoses. Condition
groups are broad synthetic buckets only.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from . import config as C


# ---------------------------------------------------------------------------
# Dimension / reference tables
# ---------------------------------------------------------------------------
def make_payers(cfg: dict[str, Any]) -> pd.DataFrame:
    """Return the payer dimension table from config."""
    df = pd.DataFrame(cfg["payers"])
    return df[["payer_id", "payer_name", "payer_type", "base_pmpm", "annual_trend"]]


def make_provider_orgs(cfg: dict[str, Any]) -> pd.DataFrame:
    """Return the provider-organization dimension table from config."""
    df = pd.DataFrame(cfg["provider_orgs"])
    return df[
        ["provider_org_id", "provider_org_name", "region", "org_type", "org_trend_adj"]
    ]


def make_service_categories(cfg: dict[str, Any]) -> pd.DataFrame:
    """Return the service-category reference table from config."""
    df = pd.DataFrame(cfg["service_categories"])
    return df[
        [
            "service_category",
            "category_group",
            "share",
            "mean_allowed",
            "extra_trend",
            "util_per_1k_tracked",
        ]
    ]


def make_condition_groups(cfg: dict[str, Any]) -> pd.DataFrame:
    """Return the (broad, synthetic) condition-group reference table."""
    return pd.DataFrame({"condition_group": cfg["condition_groups"]})


# ---------------------------------------------------------------------------
# Enrollment (member-month grain)
# ---------------------------------------------------------------------------
# Maps a payer's plan type to the line of business reported for its members.
_PAYER_TYPE_TO_LOB = {
    "Commercial": "Commercial",
    "Medicaid": "Medicaid",
    "Medicare": "Medicare Advantage",
}


def generate_enrollment(
    cfg: dict[str, Any],
    payers: pd.DataFrame,
    provider_orgs: pd.DataFrame,
    rng: np.random.Generator,
) -> pd.DataFrame:
    """Generate a member-month enrollment table.

    Each member is assigned a payer, an attributed provider organization, a
    line of business (derived from payer type), an age band and a region
    (inherited from the provider org). Members are enrolled across a span of
    months; a configurable fraction receive a mid-panel enrollment gap so the
    validation engine has real gaps to detect.
    """
    n_members = int(cfg["n_members"])
    years = list(cfg["years"])

    # --- build the month spine (one row per calendar month in the window) ---
    months = pd.period_range(
        start=f"{years[0]}-01", end=f"{years[-1]}-12", freq="M"
    )
    month_spine = pd.DataFrame(
        {
            "month_idx": np.arange(len(months)),
            "enrollment_month": months.astype(str),
            "year": months.year,
        }
    )
    n_months = len(months)

    # --- assign each member to a payer / provider org / demographics ---
    payer_ids = payers["payer_id"].to_numpy()
    member_payer = rng.choice(payer_ids, size=n_members)

    org_ids = provider_orgs["provider_org_id"].to_numpy()
    member_org = rng.choice(org_ids, size=n_members)

    age_bands = np.array(cfg["age_bands"])
    member_age = rng.choice(age_bands, size=n_members)

    members = pd.DataFrame(
        {
            "member_id": [f"MBR{n:06d}" for n in range(1, n_members + 1)],
            "payer_id": member_payer,
            "provider_org_id": member_org,
            "age_band": member_age,
        }
    )

    # Line of business follows the payer's plan type; region follows the org.
    members = members.merge(
        payers[["payer_id", "payer_type"]], on="payer_id", how="left"
    )
    members["line_of_business"] = members["payer_type"].map(_PAYER_TYPE_TO_LOB)
    members = members.merge(
        provider_orgs[["provider_org_id", "region"]],
        on="provider_org_id",
        how="left",
    )

    # --- enrollment span: most members full-panel, some partial ---
    start_idx = np.zeros(n_members, dtype=int)
    late_join = rng.random(n_members) < 0.15
    start_idx[late_join] = rng.integers(1, n_months - 6, size=late_join.sum())

    end_idx = np.full(n_members, n_months - 1, dtype=int)
    early_leave = rng.random(n_members) < 0.10
    # leavers exit somewhere after their start but before the final month
    end_idx[early_leave] = rng.integers(
        6, n_months - 1, size=early_leave.sum()
    )
    end_idx = np.maximum(end_idx, start_idx)  # guard against start > end

    # --- enrollment gaps: a single removed mid-span month for some members ---
    gap_month = np.full(n_members, -1, dtype=int)
    has_gap = rng.random(n_members) < float(cfg["defects"]["enrollment_gap_fraction"])
    span = end_idx - start_idx
    eligible = has_gap & (span >= 4)  # need room for an interior gap
    # place the gap roughly mid-span
    gap_month[eligible] = start_idx[eligible] + (span[eligible] // 2)

    members["start_idx"] = start_idx
    members["end_idx"] = end_idx
    members["gap_month"] = gap_month

    # --- expand members x months, then trim to each member's span ---
    enroll = members.merge(month_spine, how="cross")
    keep = (
        (enroll["month_idx"] >= enroll["start_idx"])
        & (enroll["month_idx"] <= enroll["end_idx"])
        & (enroll["month_idx"] != enroll["gap_month"])
    )
    enroll = enroll.loc[keep].reset_index(drop=True)

    cols = [
        "member_id",
        "payer_id",
        "provider_org_id",
        "enrollment_month",
        "year",
        "month_idx",
        "age_band",
        "region",
        "line_of_business",
    ]
    return enroll[cols]


# ---------------------------------------------------------------------------
# Claims (claim grain)
# ---------------------------------------------------------------------------
def _combined_member_trend(
    enroll: pd.DataFrame, payers: pd.DataFrame, provider_orgs: pd.DataFrame
) -> pd.Series:
    """Per-member-month overall PMPM growth rate = payer trend + org adjustment."""
    merged = enroll.merge(
        payers[["payer_id", "annual_trend", "base_pmpm"]], on="payer_id", how="left"
    ).merge(
        provider_orgs[["provider_org_id", "org_trend_adj"]],
        on="provider_org_id",
        how="left",
    )
    return merged


def generate_claims(
    cfg: dict[str, Any],
    enroll: pd.DataFrame,
    payers: pd.DataFrame,
    provider_orgs: pd.DataFrame,
    service_categories: pd.DataFrame,
    rng: np.random.Generator,
) -> pd.DataFrame:
    """Generate claim-level records tied to enrolled member-months.

    Cost growth is split between utilization (claim frequency) and unit price
    so that:

    * overall PMPM for a member grows at ``payer.annual_trend + org_trend_adj``
      (its "combined trend"), and
    * service-category mix shifts over time — categories with a higher
      ``extra_trend`` (pharmacy, behavioral health) grow faster than average,
      making them identifiable cost drivers — without distorting the member's
      overall growth.
    """
    cats = service_categories
    cat_names = cats["service_category"].to_numpy()
    shares = cats["share"].to_numpy(dtype=float)
    shares = shares / shares.sum()  # normalize to a valid probability vector
    mean_allowed = dict(zip(cats["service_category"], cats["mean_allowed"]))
    extra_trend = dict(zip(cats["service_category"], cats["extra_trend"]))

    # Share-weighted average per-claim allowed (sets claim frequency).
    weighted_mean_allowed = float((shares * cats["mean_allowed"].to_numpy()).sum())
    # DOLLAR-weighted average extra trend. The category redistribution below
    # must be neutral in DOLLARS (not claim volume) or a low-trend, high-dollar
    # category like Inpatient would drag overall PMPM growth away from the
    # seeded payer trend. Weight each category's extra_trend by its expected
    # share of spend (share * mean_allowed), not by its share of claim counts.
    dollar_weight = (shares * cats["mean_allowed"].to_numpy())
    dollar_weight = dollar_weight / dollar_weight.sum()
    avg_extra = float((dollar_weight * cats["extra_trend"].to_numpy()).sum())

    # Enrollment enriched with the member's economic parameters.
    m = _combined_member_trend(enroll, payers, provider_orgs)
    combined_trend = (m["annual_trend"] + m["org_trend_adj"]).to_numpy()
    base_pmpm = m["base_pmpm"].to_numpy()
    year = m["year"].to_numpy()
    yi = year - int(cfg["years"][0])  # 0-based year index

    # Base claim frequency per member-month implied by the payer's PMPM level.
    # Half the combined trend flows to utilization (frequency), half to price.
    lambda0 = base_pmpm / weighted_mean_allowed
    freq_factor = (1.0 + combined_trend / 2.0) ** yi
    lam = lambda0 * freq_factor

    n_claims = rng.poisson(lam)
    total_claims = int(n_claims.sum())

    # Repeat each member-month row once per claim it generates.
    rep = np.repeat(np.arange(len(m)), n_claims)
    claims = pd.DataFrame(
        {
            "member_id": m["member_id"].to_numpy()[rep],
            "payer_id": m["payer_id"].to_numpy()[rep],
            "provider_org_id": m["provider_org_id"].to_numpy()[rep],
            "line_of_business": m["line_of_business"].to_numpy()[rep],
            "region": m["region"].to_numpy()[rep],
            "enrollment_month": m["enrollment_month"].to_numpy()[rep],
        }
    )
    yi_rep = yi[rep]
    combined_rep = combined_trend[rep]

    # --- assign a service category to each claim ---
    cat_idx = rng.choice(len(cat_names), size=total_claims, p=shares)
    claims["service_category"] = cat_names[cat_idx]

    # --- unit price: base mean grown by (price half-trend) and the category's
    #     deviation from the average extra trend (redistributes the mix) ---
    base_amt = cats["mean_allowed"].to_numpy()[cat_idx]
    cat_extra = cats["extra_trend"].to_numpy()[cat_idx]
    price_factor = (1.0 + combined_rep / 2.0) ** yi_rep
    redistribution = (1.0 + (cat_extra - avg_extra)) ** yi_rep
    # log-normal multiplicative noise, mean-corrected to ~1.0. Kept moderate so
    # the per-claim cost distribution is realistically right-skewed without an
    # inpatient tail so heavy that payer-level YoY trend becomes pure noise.
    sigma = 0.35
    noise = rng.lognormal(mean=-(sigma**2) / 2.0, sigma=sigma, size=total_claims)
    allowed = base_amt * price_factor * redistribution * noise
    claims["allowed_amount"] = np.round(allowed, 2)

    # --- claim status: mostly paid, some denied / pending ---
    status_roll = rng.random(total_claims)
    status = np.where(status_roll < 0.04, "Denied",
              np.where(status_roll < 0.07, "Pending", "Paid"))
    claims["claim_status"] = status

    # --- member cost share & paid amount (paid <= allowed by construction) ---
    cs_rate_map = cfg["cost_share_rate"]
    cs_rate = claims["line_of_business"].map(cs_rate_map).to_numpy()
    # add mild noise to the cost-share fraction, clip to a sane range
    cs_rate = np.clip(cs_rate * rng.normal(1.0, 0.15, total_claims), 0.0, 0.6)
    member_cost_share = np.round(claims["allowed_amount"].to_numpy() * cs_rate, 2)
    paid = np.round(claims["allowed_amount"].to_numpy() - member_cost_share, 2)
    # denied claims pay nothing; the full allowed becomes member responsibility
    denied = claims["claim_status"].to_numpy() == "Denied"
    paid[denied] = 0.0
    member_cost_share[denied] = claims["allowed_amount"].to_numpy()[denied]
    claims["paid_amount"] = paid
    claims["member_cost_share"] = member_cost_share

    # --- service & received dates ---
    month_periods = pd.PeriodIndex(claims["enrollment_month"].to_numpy(), freq="M")
    days_in_month = month_periods.days_in_month.to_numpy()
    day = (rng.random(total_claims) * days_in_month).astype(int) + 1
    service_date = pd.to_datetime(
        dict(
            year=month_periods.year,
            month=month_periods.month,
            day=day,
        )
    )
    claims["service_date"] = service_date
    lag = rng.integers(3, 75, size=total_claims)  # claim receipt lag in days
    claims["received_date"] = service_date + pd.to_timedelta(lag, unit="D")

    # --- broad synthetic condition group ---
    cond = np.array(cfg["condition_groups"])
    claims["condition_group"] = rng.choice(cond, size=total_claims)

    # --- stable, unique claim id ---
    claims.insert(0, "claim_id", [f"CLM{n:08d}" for n in range(1, total_claims + 1)])

    claims = _add_high_cost_claimants(cfg, claims, enroll, rng)

    col_order = [
        "claim_id",
        "member_id",
        "payer_id",
        "provider_org_id",
        "line_of_business",
        "region",
        "service_date",
        "enrollment_month",
        "service_category",
        "condition_group",
        "allowed_amount",
        "paid_amount",
        "member_cost_share",
        "claim_status",
        "received_date",
    ]
    return claims[col_order].reset_index(drop=True)


def _add_high_cost_claimants(
    cfg: dict[str, Any],
    claims: pd.DataFrame,
    enroll: pd.DataFrame,
    rng: np.random.Generator,
) -> pd.DataFrame:
    """Append catastrophic claims for a small fraction of members.

    Supports high-cost-claimant concentration analysis: ~2% of members get one
    very large inpatient/oncology claim, so a small share of members account
    for a disproportionate share of total spend (as in real populations).
    """
    hc = cfg["high_cost"]
    members = enroll["member_id"].drop_duplicates().to_numpy()
    n_hc = max(1, int(len(members) * float(hc["member_fraction"])))
    chosen = rng.choice(members, size=n_hc, replace=False)

    # Attach each chosen member to ONE randomly chosen enrolled month. Spreading
    # the catastrophic claim uniformly across the member's enrolled period (vs.
    # always the last month) keeps high-cost spend roughly proportional to
    # member-months per year, so it does not distort the seeded YoY trend.
    member_pool = enroll[enroll["member_id"].isin(chosen)]
    member_month = member_pool.sample(
        frac=1.0, random_state=int(rng.integers(1e9))
    ).drop_duplicates(subset="member_id")[
        ["member_id", "payer_id", "provider_org_id",
         "line_of_business", "region", "enrollment_month"]
    ].reset_index(drop=True)

    n = len(member_month)
    allowed = rng.uniform(
        float(hc["catastrophic_allowed_min"]),
        float(hc["catastrophic_allowed_max"]),
        size=n,
    ).round(2)
    cs_rate_map = cfg["cost_share_rate"]
    cs_rate = member_month["line_of_business"].map(cs_rate_map).to_numpy()
    # high-cost claims usually blow through out-of-pocket maximums -> low share
    member_cost_share = np.minimum(allowed * cs_rate, 6000.0).round(2)
    paid = (allowed - member_cost_share).round(2)

    month_periods = pd.PeriodIndex(
        member_month["enrollment_month"].to_numpy(), freq="M"
    )
    day = (rng.random(n) * month_periods.days_in_month.to_numpy()).astype(int) + 1
    service_date = pd.to_datetime(
        dict(year=month_periods.year, month=month_periods.month, day=day)
    )

    start = int(claims["claim_id"].str.slice(3).astype(int).max()) + 1
    hc_claims = pd.DataFrame(
        {
            "claim_id": [f"CLM{start + i:08d}" for i in range(n)],
            "member_id": member_month["member_id"].to_numpy(),
            "payer_id": member_month["payer_id"].to_numpy(),
            "provider_org_id": member_month["provider_org_id"].to_numpy(),
            "line_of_business": member_month["line_of_business"].to_numpy(),
            "region": member_month["region"].to_numpy(),
            "service_date": service_date,
            "enrollment_month": member_month["enrollment_month"].to_numpy(),
            "service_category": rng.choice(
                ["Inpatient", "Outpatient"], size=n, p=[0.8, 0.2]
            ),
            "condition_group": rng.choice(["Oncology", "Cardiometabolic"], size=n),
            "allowed_amount": allowed,
            "paid_amount": paid,
            "member_cost_share": member_cost_share,
            "claim_status": np.array(["Paid"] * n),
            "received_date": service_date + pd.to_timedelta(
                rng.integers(10, 90, size=n), unit="D"
            ),
        }
    )
    return pd.concat([claims, hc_claims], ignore_index=True)


def concentrate_high_cost_facility(
    cfg: dict[str, Any],
    claims: pd.DataFrame,
    enrollment: pd.DataFrame,
    provider_orgs: pd.DataFrame,
    rng: np.random.Generator,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Re-attribute a share of high-cost members to a tertiary/specialty facility.

    Models a referral center with a disproportionately sick panel: a configured
    fraction of members who incurred a catastrophic claim are moved (enrollment
    and claims together, with their region) to one provider org. Run BEFORE
    defect injection so it doesn't disturb the seeded missing-provider defects.
    """
    hc = cfg["high_cost"]
    org_id = hc.get("specialty_org_id")
    share = float(hc.get("specialty_share", 0.0) or 0.0)
    if not org_id or share <= 0:
        return claims, enrollment

    trunc = float(cfg["truncation_threshold"])
    hc_members = claims.loc[claims["allowed_amount"] > trunc, "member_id"].unique()
    n_move = int(len(hc_members) * share)
    if n_move <= 0:
        return claims, enrollment

    move = set(rng.choice(hc_members, size=n_move, replace=False))
    region = provider_orgs.loc[
        provider_orgs["provider_org_id"] == org_id, "region"
    ].iloc[0]

    e_mask = enrollment["member_id"].isin(move)
    enrollment.loc[e_mask, "provider_org_id"] = org_id
    enrollment.loc[e_mask, "region"] = region
    c_mask = claims["member_id"].isin(move)
    claims.loc[c_mask, "provider_org_id"] = org_id
    claims.loc[c_mask, "region"] = region
    return claims, enrollment


# ---------------------------------------------------------------------------
# Data-quality defect injection (run LAST, on an otherwise clean dataset)
# ---------------------------------------------------------------------------
def inject_quality_defects(
    cfg: dict[str, Any],
    claims: pd.DataFrame,
    enroll: pd.DataFrame,
    rng: np.random.Generator,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Inject a controlled set of known data-quality defects into the claims.

    Returns the (claims, enrollment) pair. Each defect corresponds to a check
    the Phase 2 validation engine will perform, so the validation report has
    real, countable findings. Defect rates are configured in ``config.yaml``.
    """
    d = cfg["defects"]
    claims = claims.copy()
    n = len(claims)

    def sample(rate: float) -> np.ndarray:
        """Random row positions for a given defect rate."""
        k = int(n * float(rate))
        if k <= 0:
            return np.array([], dtype=int)
        return rng.choice(n, size=k, replace=False)

    # 1) Negative allowed/paid amounts ------------------------------------
    idx = sample(d["negative_amount_rate"])
    claims.loc[claims.index[idx], "allowed_amount"] *= -1
    claims.loc[claims.index[idx], "paid_amount"] *= -1

    # 2) Paid > allowed ----------------------------------------------------
    idx = sample(d["paid_gt_allowed_rate"])
    claims.loc[claims.index[idx], "paid_amount"] = (
        claims.loc[claims.index[idx], "allowed_amount"].abs() * 1.25
    ).round(2)

    # 3) Missing service category -----------------------------------------
    idx = sample(d["missing_service_cat_rate"])
    claims.loc[claims.index[idx], "service_category"] = np.nan

    # 4) Missing provider org id ------------------------------------------
    idx = sample(d["missing_provider_rate"])
    claims.loc[claims.index[idx], "provider_org_id"] = np.nan

    # 5) Invalid service dates (outside the measurement window) -----------
    idx = sample(d["invalid_date_rate"])
    bad_year = int(cfg["years"][0]) - 2
    claims.loc[claims.index[idx], "service_date"] = pd.Timestamp(
        f"{bad_year}-06-15"
    )

    # 6) Orphan claims: members that never appear in enrollment -----------
    k_orphan = int(n * float(d["orphan_claim_rate"]))
    if k_orphan > 0:
        template = claims.sample(k_orphan, random_state=int(rng.integers(1e9))).copy()
        template["member_id"] = [f"MBR9{i:06d}" for i in range(k_orphan)]
        start = int(claims["claim_id"].str.slice(3).astype(int).max()) + 1
        template["claim_id"] = [f"CLM{start + i:08d}" for i in range(k_orphan)]
        claims = pd.concat([claims, template], ignore_index=True)

    # 7) Exact duplicate claim rows (same claim_id) -----------------------
    k_dup = int(len(claims) * float(d["duplicate_claim_rate"]))
    if k_dup > 0:
        dup = claims.sample(k_dup, random_state=int(rng.integers(1e9))).copy()
        claims = pd.concat([claims, dup], ignore_index=True)

    # 8) Incomplete submission: drop most of one payer/year's claims ------
    inc = d["incomplete_submission"]
    mask = (claims["payer_id"] == inc["payer_id"]) & (
        pd.to_datetime(claims["service_date"]).dt.year == int(inc["year"])
    )
    seg = claims[mask]
    if len(seg) > 0:
        keep_n = int(len(seg) * float(inc["keep_fraction"]))
        drop_idx = seg.sample(len(seg) - keep_n, random_state=int(rng.integers(1e9))).index
        claims = claims.drop(index=drop_idx).reset_index(drop=True)

    return claims, enroll


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------
def build_all(cfg: dict[str, Any] | None = None) -> dict[str, pd.DataFrame]:
    """Generate every source table and return them keyed by short name."""
    if cfg is None:
        cfg = C.load_config()
    rng = np.random.default_rng(int(cfg["random_seed"]))

    payers = make_payers(cfg)
    provider_orgs = make_provider_orgs(cfg)
    service_categories = make_service_categories(cfg)
    condition_groups = make_condition_groups(cfg)

    enrollment = generate_enrollment(cfg, payers, provider_orgs, rng)
    claims = generate_claims(
        cfg, enrollment, payers, provider_orgs, service_categories, rng
    )
    claims, enrollment = concentrate_high_cost_facility(
        cfg, claims, enrollment, provider_orgs, rng
    )
    claims, enrollment = inject_quality_defects(cfg, claims, enrollment, rng)

    # drop the internal helper column before writing the public table
    enrollment_out = enrollment.drop(columns=["month_idx"], errors="ignore")

    return {
        "payer": payers,
        "provider_org": provider_orgs,
        "service_category": service_categories,
        "condition": condition_groups,
        "enrollment": enrollment_out,
        "claims": claims,
    }


def write_raw(tables: dict[str, pd.DataFrame]) -> dict[str, str]:
    """Write each table to data/raw as CSV. Returns name -> path written."""
    C.ensure_dirs()
    written: dict[str, str] = {}
    for key, df in tables.items():
        fname = C.RAW_FILES[key]
        path = C.RAW_DIR / fname
        df.to_csv(path, index=False)
        written[key] = str(path)
    return written


def main() -> None:
    cfg = C.load_config()
    tables = build_all(cfg)
    written = write_raw(tables)
    print("Synthetic source tables written to data/raw:\n")
    for key, df in tables.items():
        print(f"  {C.RAW_FILES[key]:32s} {len(df):>9,d} rows  -> {written[key]}")
    n_claims = len(tables["claims"])
    n_mm = len(tables["enrollment"])
    print(f"\n  member-months: {n_mm:,d}   claims: {n_claims:,d}")


if __name__ == "__main__":
    main()
