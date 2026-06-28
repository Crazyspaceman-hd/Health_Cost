"""Cost and utilization metrics engine.

Turns the validated source tables into the headline metrics of a cost-growth
program:

* member months (the PMPM denominator)
* total medical expense (allowed), total paid, member cost share
* PMPM — reported both **raw** and **high-cost truncated**
* year-over-year PMPM growth and growth **versus the 3.4% target**
* the same growth cut by payer, provider organization, line of business and
  service category
* utilization per 1,000 member-months by service category
* high-cost member / claim spend concentration

Metrics are computed on an **analytic claims** set: the raw claims after the
Phase 2 data-quality defects are removed (de-duplicated, non-negative, valid
dates, attributable to an enrolled member). The deliberately *incomplete*
PAY002 2023 submission is intentionally retained — it is valid data that is
merely incomplete, and the resulting trend artifact is surfaced in reporting.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from . import config as C
from . import data_io
from . import validate as V


# ---------------------------------------------------------------------------
# Completeness adjustment: drop validation-flagged incomplete submissions
# ---------------------------------------------------------------------------
def incomplete_payer_years(
    data: dict[str, pd.DataFrame], cfg: dict[str, Any]
) -> set[tuple[str, int]]:
    """(payer_id, year) cells flagged as incomplete submissions by check DQ013."""
    res = V.check_yoy_volume_change(data, cfg)
    cells: set[tuple[str, int]] = set()
    for eid in res.details["entity_id"]:
        payer, year = str(eid).split("|")
        cells.add((payer, int(year)))
    return cells


def drop_payers(
    data: dict[str, pd.DataFrame], payers: set[str]
) -> dict[str, pd.DataFrame]:
    """Remove entire payers (all years) from claims and enrollment.

    Used for *pooled* trends (program / provider / category), which must be
    measured on a composition-consistent panel of complete submitters. Dropping
    an incomplete payer from only one year would bias the pooled year-over-year
    mix; dropping it from every year keeps the panel stable.
    """
    if not payers:
        return data
    out = dict(data)
    out["claims"] = data["claims"][~data["claims"]["payer_id"].isin(payers)].copy()
    out["enrollment"] = data["enrollment"][
        ~data["enrollment"]["payer_id"].isin(payers)
    ].copy()
    return out


def apply_completeness_filter(
    data: dict[str, pd.DataFrame], cells: set[tuple[str, int]]
) -> dict[str, pd.DataFrame]:
    """Remove flagged payer-years from BOTH claims and enrollment.

    Dropping the cell from the numerator (claims) and the denominator
    (member-months) together means the incomplete payer-year is simply absent
    from the trend, rather than depressing PMPM. This is the standard way a
    cost-growth program handles a submission it has deemed incomplete.
    """
    if not cells:
        return data
    claims = data["claims"]
    enr = data["enrollment"]
    claim_year = pd.to_datetime(claims["service_date"], errors="coerce").dt.year
    claim_cells = pd.Series(list(zip(claims["payer_id"], claim_year)),
                            index=claims.index)
    enr_cells = pd.Series(list(zip(enr["payer_id"], enr["year"])), index=enr.index)
    out = dict(data)
    out["claims"] = claims[~claim_cells.isin(cells)].copy()
    out["enrollment"] = enr[~enr_cells.isin(cells)].copy()
    return out


# ---------------------------------------------------------------------------
# Analytic base: clean the claims before measuring cost
# ---------------------------------------------------------------------------
def prepare_analytic_claims(
    data: dict[str, pd.DataFrame], cfg: dict[str, Any]
) -> pd.DataFrame:
    """Return claims suitable for cost measurement, with defects removed.

    Exclusions mirror the Phase 2 validation findings:
      * exact duplicate claim_ids -> keep first occurrence
      * negative allowed/paid amounts -> drop
      * paid > allowed (impossible adjudication) -> drop
      * orphan claims (member not in enrollment) -> drop
      * invalid service dates (outside the measurement window / null) -> drop
    Missing service categories are retained but relabeled 'Unclassified' so the
    spend still counts toward totals; missing provider IDs are retained (they
    are simply excluded from provider-level cuts via group-by dropna).
    """
    c = data["claims"].copy()
    years = sorted(cfg["years"])
    lo, hi = pd.Timestamp(f"{years[0]}-01-01"), pd.Timestamp(f"{years[-1]}-12-31")

    c = c.drop_duplicates(subset="claim_id", keep="first")

    c["service_date"] = pd.to_datetime(c["service_date"], errors="coerce")
    valid_date = c["service_date"].between(lo, hi)

    non_negative = (c["allowed_amount"] >= 0) & (c["paid_amount"] >= 0)
    paid_ok = c["paid_amount"] <= c["allowed_amount"] + 0.01

    enrolled = set(data["enrollment"]["member_id"].unique())
    attributable = c["member_id"].isin(enrolled)

    c = c[valid_date & non_negative & paid_ok & attributable].copy()
    c["service_category"] = c["service_category"].fillna("Unclassified")
    c["year"] = c["service_date"].dt.year
    return c.reset_index(drop=True)


# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------
def member_months(enroll: pd.DataFrame, dims: list[str]) -> pd.Series:
    """Member-month counts grouped by ``dims`` (the PMPM denominator)."""
    return enroll.groupby(dims).size().rename("member_months")


def _spend(claims: pd.DataFrame, dims: list[str], trunc: float) -> pd.DataFrame:
    """Aggregate spend and claim counts by ``dims``.

    ``allowed_trunc`` winsorizes each claim's allowed amount at the high-cost
    truncation threshold so a few catastrophic claims do not dominate the trend.
    """
    c = claims.copy()
    c["allowed_trunc"] = np.minimum(c["allowed_amount"], trunc)
    g = c.groupby(dims)
    out = g.agg(
        total_allowed=("allowed_amount", "sum"),
        total_paid=("paid_amount", "sum"),
        member_cost_share=("member_cost_share", "sum"),
        total_allowed_trunc=("allowed_trunc", "sum"),
        claims=("claim_id", "size"),
    )
    return out


def _annualized_growth(values: pd.Series) -> float:
    """CAGR across the first and last available period of a year-indexed series."""
    v = values.dropna()
    if len(v) < 2 or v.iloc[0] <= 0:
        return np.nan
    n = v.index[-1] - v.index[0]
    return (v.iloc[-1] / v.iloc[0]) ** (1 / n) - 1


def cost_growth_by(
    claims: pd.DataFrame,
    enroll: pd.DataFrame,
    entity: str,
    cfg: dict[str, Any],
) -> pd.DataFrame:
    """Cost-growth table for one grouping entity (payer/provider/LOB).

    Returns one row per (entity, year) with PMPM levels, year-over-year
    truncated-PMPM growth, the gap versus target, and the entity's full-period
    CAGR (repeated on each of its rows for convenient filtering).
    """
    trunc = float(cfg["truncation_threshold"])
    target = float(cfg["target_growth_rate"])

    spend = _spend(claims, [entity, "year"], trunc)
    mm = member_months(enroll, [entity, "year"])
    df = spend.join(mm, how="left").reset_index()
    df = df[df["member_months"].notna() & (df["member_months"] > 0)].copy()

    df["allowed_pmpm"] = df["total_allowed"] / df["member_months"]
    df["paid_pmpm"] = df["total_paid"] / df["member_months"]
    df["truncated_pmpm"] = df["total_allowed_trunc"] / df["member_months"]
    df = df.sort_values([entity, "year"])

    # year-over-year growth on the truncated PMPM (the program's trend measure)
    df["prior_truncated_pmpm"] = df.groupby(entity)["truncated_pmpm"].shift(1)
    df["yoy_growth"] = df["truncated_pmpm"] / df["prior_truncated_pmpm"] - 1
    df["vs_target"] = df["yoy_growth"] - target
    df["exceeds_target"] = df["yoy_growth"] > target

    # full-period CAGR per entity (same value repeated across its years)
    cagr = (
        df.set_index("year")
        .groupby(entity)["truncated_pmpm"]
        .apply(_annualized_growth)
        .rename("cagr_truncated")
    )
    df = df.merge(cagr, on=entity, how="left")
    df["cagr_exceeds_target"] = df["cagr_truncated"] > target
    return df.reset_index(drop=True)


def service_category_trends(
    claims: pd.DataFrame, enroll: pd.DataFrame, cfg: dict[str, Any]
) -> pd.DataFrame:
    """Per-category, per-year PMPM contribution and utilization per 1,000 MM.

    The denominator for a category is the **total** member-months for the year
    (every member-month is exposed to every category), so category PMPMs sum to
    the overall PMPM.
    """
    trunc = float(cfg["truncation_threshold"])
    target = float(cfg["target_growth_rate"])

    spend = _spend(claims, ["service_category", "year"], trunc).reset_index()
    total_mm = member_months(enroll, ["year"])  # by year only

    df = spend.merge(total_mm.reset_index(), on="year", how="left")
    df["allowed_pmpm"] = df["total_allowed"] / df["member_months"]
    df["truncated_pmpm"] = df["total_allowed_trunc"] / df["member_months"]
    df["utilization_per_1000_mm"] = df["claims"] / df["member_months"] * 1000
    df = df.sort_values(["service_category", "year"])

    grp = df.groupby("service_category")
    df["yoy_pmpm_growth"] = df["truncated_pmpm"] / grp["truncated_pmpm"].shift(1) - 1
    df["yoy_util_growth"] = (
        df["utilization_per_1000_mm"] / grp["utilization_per_1000_mm"].shift(1) - 1
    )
    df["vs_target"] = df["yoy_pmpm_growth"] - target
    return df.reset_index(drop=True)


def provider_category_growth(
    claims: pd.DataFrame, enroll: pd.DataFrame, cfg: dict[str, Any]
) -> pd.DataFrame:
    """Service-category cost growth within each provider organization.

    Returns one row per (provider, service_category) with truncated PMPM in the
    two complete measurement years and the growth between them. The denominator
    for each category is the provider's TOTAL member-months, so a provider's
    category PMPMs sum to its overall PMPM. Measured over the complete years to
    avoid the incomplete-submission artifact.
    """
    trunc = float(cfg["truncation_threshold"])
    y0, y1 = sorted(cfg["years"])[:2]
    c = claims[claims["provider_org_id"].notna()]

    spend = _spend(
        c, ["provider_org_id", "service_category", "year"], trunc
    ).reset_index()
    mm = member_months(enroll, ["provider_org_id", "year"]).reset_index()
    df = spend.merge(mm, on=["provider_org_id", "year"], how="left")
    df["pmpm"] = df["total_allowed_trunc"] / df["member_months"]

    wide = (
        df[df["year"].isin([y0, y1])]
        .pivot_table(index=["provider_org_id", "service_category"],
                     columns="year", values="pmpm")
        .reset_index()
    )
    wide = wide.rename(columns={y0: "pmpm_start", y1: "pmpm_end"})
    wide["growth"] = wide["pmpm_end"] / wide["pmpm_start"] - 1
    return wide


def price_utilization_decomposition(
    claims: pd.DataFrame, enroll: pd.DataFrame, cfg: dict[str, Any]
) -> pd.DataFrame:
    """Split each service category's PMPM growth into price vs. utilization.

    PMPM = utilization x unit price, where
        utilization = claims / member-month   (volume)
        unit price  = truncated allowed $ / claim   (cost per service)
    so PMPM growth decomposes into a utilization rate and a unit-price rate
    (they sum, up to a small interaction term, to PMPM growth). This answers
    whether a category's cost growth is driven by *more services* or by *each
    service costing more*. Measured over the two complete years.

    ``util_pts`` + ``price_pts`` sum exactly to ``pmpm_growth`` (the
    utilization x price interaction is assigned to price).
    """
    trunc = float(cfg["truncation_threshold"])
    y0, y1 = sorted(cfg["years"])[:2]
    c = claims.copy()
    c["allowed_trunc"] = np.minimum(c["allowed_amount"], trunc)
    mm = enroll.groupby("year").size()

    g = (
        c.groupby(["service_category", "year"])
        .agg(n=("claim_id", "size"), allowed=("allowed_trunc", "sum"))
        .reset_index()
    )
    g["mm"] = g["year"].map(mm)
    g["util"] = g["n"] / g["mm"]              # services per member-month
    g["price"] = g["allowed"] / g["n"]        # $ per service
    g["pmpm"] = g["allowed"] / g["mm"]

    rows = []
    for cat, sub in g.groupby("service_category"):
        s = sub.set_index("year")
        if y0 not in s.index or y1 not in s.index:
            continue
        gu = s.loc[y1, "util"] / s.loc[y0, "util"] - 1
        gp = s.loc[y1, "price"] / s.loc[y0, "price"] - 1
        gpmpm = s.loc[y1, "pmpm"] / s.loc[y0, "pmpm"] - 1
        rows.append({
            "service_category": cat,
            "util_growth": gu,
            "price_growth": gp,
            "pmpm_growth": gpmpm,
            "util_pts": gu,                  # utilization contribution
            "price_pts": gpmpm - gu,         # price contribution (incl. interaction)
        })
    return pd.DataFrame(rows)


def high_cost_by_facility(
    claims: pd.DataFrame, enroll: pd.DataFrame, cfg: dict[str, Any]
) -> pd.DataFrame:
    """High-cost member concentration and case-mix effect by provider org.

    A member is "high-cost" if their total allowed spend exceeds the truncation
    threshold. For each facility returns its panel size, high-cost member count,
    its share of all high-cost members, a concentration index (share of
    high-cost members ÷ share of the panel; 1.0 = proportional), and raw vs
    truncated PMPM — the gap between the two shows how much of a facility's raw
    cost is catastrophic case-mix rather than underlying price/utilization.
    """
    trunc = float(cfg["truncation_threshold"])
    member_org = enroll.groupby("member_id")["provider_org_id"].first()
    spend = claims.groupby("member_id")["allowed_amount"].sum()
    m = pd.DataFrame({"provider_org_id": member_org, "spend": spend}).dropna()
    m["is_hc"] = m["spend"] > trunc
    total_members, total_hc = len(m), int(m["is_hc"].sum())

    mm = enroll.groupby("provider_org_id").size()
    c = claims.copy()
    c["allowed_trunc"] = np.minimum(c["allowed_amount"], trunc)
    raw = c.groupby("provider_org_id")["allowed_amount"].sum()
    tru = c.groupby("provider_org_id")["allowed_trunc"].sum()

    rows = []
    for org, sub in m.groupby("provider_org_id"):
        n, nhc = len(sub), int(sub["is_hc"].sum())
        panel_share = n / total_members
        hc_share = nhc / total_hc if total_hc else float("nan")
        rows.append({
            "provider_org_id": org,
            "members": n,
            "high_cost_members": nhc,
            "panel_share": panel_share,
            "high_cost_share": hc_share,
            "concentration_index": (hc_share / panel_share)
            if panel_share else float("nan"),
            "raw_pmpm": raw.get(org, 0.0) / mm.get(org, float("nan")),
            "truncated_pmpm": tru.get(org, 0.0) / mm.get(org, float("nan")),
        })
    out = pd.DataFrame(rows)
    out["case_mix_load"] = out["raw_pmpm"] / out["truncated_pmpm"]
    return out


def provider_growth_profile(
    claims: pd.DataFrame, enroll: pd.DataFrame, cfg: dict[str, Any]
) -> pd.DataFrame:
    """Characterize each provider's cost growth as broad-based vs concentrated.

    For every provider returns:
      * ``n_above_target`` — how many service categories grew faster than the
        program target (breadth of the growth);
      * ``n_categories`` — categories evaluated;
      * ``top_category`` / ``top_category_dollar_share`` — the single category
        contributing the most PMPM growth dollars and its share of the
        provider's total *positive* PMPM growth (concentration of the dollars).

    Breadth uses growth rates; the dollar share weights by category size, so the
    two together separate broad systemic growth from growth that is large only
    because one big category (typically inpatient) moved.
    """
    target = float(cfg["target_growth_rate"])
    pc = provider_category_growth(claims, enroll, cfg)
    pc = pc[pc["service_category"] != "Unclassified"].copy()
    pc["delta"] = pc["pmpm_end"] - pc["pmpm_start"]

    rows = []
    for pid, sub in pc.groupby("provider_org_id"):
        n_cat = int(sub["service_category"].nunique())
        n_above = int((sub["growth"] > target).sum())
        pos = sub[sub["delta"] > 0]
        tot_pos = pos["delta"].sum()
        if len(pos) and tot_pos > 0:
            top = pos.loc[pos["delta"].idxmax()]
            top_cat = top["service_category"]
            top_share = float(top["delta"] / tot_pos)
        else:
            top_cat, top_share = None, float("nan")
        rows.append({
            "provider_org_id": pid,
            "n_above_target": n_above,
            "n_categories": n_cat,
            "top_category": top_cat,
            "top_category_dollar_share": top_share,
        })
    return pd.DataFrame(rows)


def high_cost_concentration(
    claims: pd.DataFrame, cfg: dict[str, Any]
) -> pd.DataFrame:
    """Spend concentration: share of total allowed held by the top X% of members.

    Reported per year and for all years pooled. Also reports the share of spend
    from claims above the high-cost truncation threshold.
    """
    trunc = float(cfg["truncation_threshold"])
    rows = []

    def _row(label: str, c: pd.DataFrame) -> dict:
        by_member = c.groupby("member_id")["allowed_amount"].sum().sort_values(
            ascending=False
        )
        total = by_member.sum()
        out = {"period": label, "n_members": len(by_member),
               "total_allowed": total}
        for pct in (0.01, 0.05, 0.10):
            k = max(1, int(len(by_member) * pct))
            out[f"top_{int(pct*100)}pct_spend_share"] = (
                by_member.head(k).sum() / total if total else np.nan
            )
        hc_spend = c.loc[c["allowed_amount"] > trunc, "allowed_amount"].sum()
        out["high_cost_claim_spend_share"] = hc_spend / total if total else np.nan
        return out

    for yr, c in claims.groupby("year"):
        rows.append(_row(str(yr), c))
    rows.append(_row("All", claims))
    return pd.DataFrame(rows)


def executive_summary_metrics(
    claims: pd.DataFrame, enroll: pd.DataFrame, cfg: dict[str, Any]
) -> pd.DataFrame:
    """Program-level headline metrics, one row per year plus growth vs target."""
    trunc = float(cfg["truncation_threshold"])
    target = float(cfg["target_growth_rate"])

    spend = _spend(claims, ["year"], trunc).reset_index()
    mm = member_months(enroll, ["year"]).reset_index()
    df = spend.merge(mm, on="year", how="left").sort_values("year")

    df["allowed_pmpm"] = df["total_allowed"] / df["member_months"]
    df["paid_pmpm"] = df["total_paid"] / df["member_months"]
    df["truncated_pmpm"] = df["total_allowed_trunc"] / df["member_months"]
    df["yoy_growth"] = df["truncated_pmpm"] / df["truncated_pmpm"].shift(1) - 1
    df["target"] = target
    df["vs_target"] = df["yoy_growth"] - target
    df["exceeds_target"] = df["yoy_growth"] > target
    df = df.rename(columns={"total_allowed": "total_medical_expense"})
    return df.reset_index(drop=True)


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------
def build_metric_tables(
    data: dict[str, pd.DataFrame],
    cfg: dict[str, Any] | None = None,
    exclude_incomplete: bool = False,
) -> dict[str, pd.DataFrame]:
    """Compute every metric table and return them keyed by output name.

    When ``exclude_incomplete`` is True, payer-years flagged as incomplete
    submissions (DQ013) are dropped from both claims and member-months before
    measurement — the completeness-adjusted view used for reporting and charts.
    """
    if cfg is None:
        cfg = C.load_config()

    # Two completeness views when excluding incomplete submissions:
    #  * payer table  -> drop only the flagged payer-YEAR (each payer is its own
    #    panel, so it can still be reported over its complete years);
    #  * pooled tables -> drop the incomplete payer for ALL years, so program/
    #    provider/category trends sit on a composition-consistent panel.
    if exclude_incomplete:
        cells = incomplete_payer_years(data, cfg)
        incomplete_p = {p for p, _ in cells}
        payer_data = apply_completeness_filter(data, cells)
        pooled_data = drop_payers(data, incomplete_p)
    else:
        cells, incomplete_p = set(), set()
        payer_data = pooled_data = data

    payer_claims = prepare_analytic_claims(payer_data, cfg)
    pooled_claims = prepare_analytic_claims(pooled_data, cfg)
    payer_enroll, pooled_enroll = payer_data["enrollment"], pooled_data["enrollment"]

    payer = cost_growth_by(payer_claims, payer_enroll, "payer_id", cfg).merge(
        data["payer"][["payer_id", "payer_name", "payer_type"]],
        on="payer_id", how="left",
    )
    # A payer with an incomplete submission has too few complete years for a
    # reliable trend; flag it so reporting can mark it "not assessed" rather
    # than classifying its noisy short-panel CAGR against the target.
    payer["complete_submission"] = ~payer["payer_id"].isin(incomplete_p)
    provider = cost_growth_by(
        pooled_claims[pooled_claims["provider_org_id"].notna()],
        pooled_enroll, "provider_org_id", cfg,
    ).merge(
        data["provider_org"][["provider_org_id", "provider_org_name", "region"]],
        on="provider_org_id", how="left",
    )
    lob = cost_growth_by(pooled_claims, pooled_enroll, "line_of_business", cfg)
    claims, enroll = pooled_claims, pooled_enroll  # pooled base for the rest

    return {
        "payer_cost_growth_summary": payer,
        "provider_cost_growth_summary": provider,
        "line_of_business_summary": lob,
        "service_category_trends": service_category_trends(claims, enroll, cfg),
        "high_cost_concentration": high_cost_concentration(claims, cfg),
        "executive_summary_metrics": executive_summary_metrics(claims, enroll, cfg),
    }


def write_metric_tables(tables: dict[str, pd.DataFrame]) -> dict[str, str]:
    """Write each metric table to outputs/ as CSV."""
    C.ensure_dirs()
    written = {}
    for name, df in tables.items():
        path = C.OUTPUT_DIR / f"{name}.csv"
        df.to_csv(path, index=False)
        written[name] = str(path)
    return written


def main() -> None:
    cfg = C.load_config()
    data = data_io.load_raw()
    tables = build_metric_tables(data, cfg)
    written = write_metric_tables(tables)

    target = cfg["target_growth_rate"]
    print(f"Metrics computed (target = {target:.1%}).\n")
    exe = tables["executive_summary_metrics"]
    cols = ["year", "member_months", "truncated_pmpm", "yoy_growth", "exceeds_target"]
    show = exe[cols].copy()
    show["truncated_pmpm"] = show["truncated_pmpm"].round(2)
    show["yoy_growth"] = (show["yoy_growth"] * 100).round(2)
    print("Program PMPM (high-cost truncated):")
    print(show.to_string(index=False))

    payer = tables["payer_cost_growth_summary"]
    over = payer.drop_duplicates("payer_id")
    n_over = int(over["cagr_exceeds_target"].sum())
    print(f"\n  Payers above target (CAGR): {n_over} of {over['payer_id'].nunique()}")
    for name, path in written.items():
        print(f"  {name:32s} -> {path}")


if __name__ == "__main__":
    main()
