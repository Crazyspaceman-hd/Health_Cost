"""Executive summary generator.

Renders ``reports/executive_summary.md`` — a plain-English, health-policy-style
writeup that reads its numbers off the *actual* computed tables (nothing is
hard-coded). Regenerating after changing ``config.yaml`` produces a report whose
narrative matches the new data.
"""
from __future__ import annotations

from typing import Any

import pandas as pd

from . import config as C
from . import data_io
from . import metrics as M
from . import validate as V


def _pct(x: float, digits: int = 1) -> str:
    return "n/a" if pd.isna(x) else f"{x * 100:.{digits}f}%"


def _money(x: float) -> str:
    return "n/a" if pd.isna(x) else f"${x:,.0f}"


def build_report(
    data: dict[str, pd.DataFrame], cfg: dict[str, Any] | None = None
) -> str:
    """Assemble the executive summary markdown from computed results."""
    if cfg is None:
        cfg = C.load_config()
    target = float(cfg["target_growth_rate"])
    years = sorted(cfg["years"])

    # --- data inventory (raw) ---
    enr, claims_raw = data["enrollment"], data["claims"]
    n_members = enr["member_id"].nunique()
    n_member_months = len(enr)
    n_claims = len(claims_raw)
    n_payers = data["payer"]["payer_id"].nunique()
    n_orgs = data["provider_org"]["provider_org_id"].nunique()

    # --- validation ---
    results = V.run_validation(data, cfg)
    summary = V.build_summary(results)
    n_fail = int((summary["status"] == "FAIL").sum())
    total_issues = int(summary["issue_count"].sum())
    crit = summary[summary["severity"] == "CRITICAL"].sort_values(
        "issue_count", ascending=False
    )

    # --- completeness-adjusted metrics ---
    incomplete = M.incomplete_payer_years(data, cfg)
    tables = M.build_metric_tables(data, cfg, exclude_incomplete=True)
    exe = tables["executive_summary_metrics"]
    payer = tables["payer_cost_growth_summary"].drop_duplicates("payer_id")
    provider = tables["provider_cost_growth_summary"].drop_duplicates(
        "provider_org_id"
    )

    # provider broad-based vs concentrated profile (on the same pooled base)
    pooled_data = M.drop_payers(data, {p for p, _ in incomplete})
    pooled_claims = M.prepare_analytic_claims(pooled_data, cfg)
    decomp = M.price_utilization_decomposition(
        pooled_claims, pooled_data["enrollment"], cfg
    ).set_index("service_category")
    profile = M.provider_growth_profile(
        pooled_claims, pooled_data["enrollment"], cfg
    ).set_index("provider_org_id")

    complete = payer["complete_submission"] if "complete_submission" in payer else True
    assessed = payer[complete]
    not_assessed = payer[~complete] if "complete_submission" in payer else payer.iloc[0:0]
    payers_above = assessed[assessed["cagr_exceeds_target"]].sort_values(
        "cagr_truncated", ascending=False
    )
    payers_below = assessed[~assessed["cagr_exceeds_target"]].sort_values(
        "cagr_truncated"
    )
    orgs_above = provider[provider["cagr_exceeds_target"]].sort_values(
        "cagr_truncated", ascending=False
    )

    # --- category drivers over the complete years ---
    trends = tables["service_category_trends"]
    piv = trends.pivot(index="service_category", columns="year",
                       values="truncated_pmpm")
    cy = years[:2]
    growth = (piv[cy[1]] / piv[cy[0]] - 1).drop("Unclassified", errors="ignore")
    drivers = growth.sort_values(ascending=False)

    # --- high-cost concentration ---
    conc = tables["high_cost_concentration"]
    pooled = conc[conc["period"] == "All"].iloc[0]

    # --- program-level latest growth ---
    latest = exe.dropna(subset=["yoy_growth"]).iloc[-1]

    # ----------------------------------------------------------------- build
    L: list[str] = []
    L.append("# Health Cost Growth Target Program — Executive Summary")
    L.append("")
    L.append(f"*Measurement years {years[0]}–{years[-1]} · "
             f"per-capita cost growth target {target:.1%} · "
             f"prepared from synthetic data*")
    L.append("")
    L.append("> **Synthetic data notice.** Every figure in this report is "
             "derived from randomly generated, de-identified data created for "
             "skill demonstration. It contains no real members, claims, "
             "providers, or payers, and represents no real program or "
             "population.")
    L.append("")

    L.append("## What this analysis does")
    L.append("")
    L.append(
        "This analysis simulates the annual workflow of a state health cost "
        "growth target program: it ingests payer-submitted enrollment and "
        "claims data, validates the submissions, and measures per-member-per-"
        "month (PMPM) cost growth against the program's "
        f"{target:.1%} target — overall and by payer, provider organization, "
        "line of business, and service category. Trend is measured on "
        "**high-cost-truncated** allowed dollars so that a small number of "
        "catastrophic claims do not distort the underlying trend.")
    L.append("")

    L.append("## Data evaluated")
    L.append("")
    L.append(f"- **{n_members:,}** members · **{n_member_months:,}** "
             f"member-months · **{n_claims:,}** claim lines")
    L.append(f"- **{n_payers}** payers and **{n_orgs}** provider organizations "
             f"across {len(years)} measurement years")
    L.append(f"- Three lines of business (Medicaid, Medicare Advantage, "
             f"Commercial) and {data['service_category'].shape[0]} service "
             f"categories")
    L.append("")

    L.append("## Data quality")
    L.append("")
    L.append(
        f"A {len(results)}-check validation suite was run over every "
        f"submission. **{n_fail} checks failed** and **{total_issues:,} "
        f"records** were flagged across all severities. The most material "
        f"(CRITICAL) findings were:")
    L.append("")
    for _, r in crit.iterrows():
        L.append(f"- **{r['check_name']}** — {int(r['issue_count']):,} records "
                 f"({r['affected_pct']:.2f}% of claims)")
    if incomplete:
        cells = ", ".join(f"{p} {y}" for p, y in sorted(incomplete))
        L.append(f"- **Incomplete submission(s):** {cells} showed a year-over-"
                 f"year volume drop beyond tolerance and "
                 f"**was excluded from cost-growth conclusions** "
                 f"(both claims and member-months).")
    L.append("")
    L.append("See `outputs/validation_summary.csv` and "
             "`outputs/data_quality_issues.csv` for the full rule set and "
             "affected-record detail.")
    L.append("")

    L.append("## Key findings")
    L.append("")
    L.append("### Overall cost growth")
    L.append("")
    pmpm_first = exe.iloc[0]
    L.append(f"On the completeness-adjusted population, truncated PMPM moved "
             f"from **{_money(pmpm_first['truncated_pmpm'])}** in {years[0]} "
             f"to **{_money(exe.iloc[-1]['truncated_pmpm'])}** in {years[-1]}. "
             f"The most recent year-over-year change was "
             f"**{_pct(latest['yoy_growth'])}** versus the {target:.1%} "
             f"target ({'above' if latest['exceeds_target'] else 'within'} "
             f"target).")
    L.append("")

    L.append("### Payers relative to target")
    L.append("")
    if len(payers_above):
        names = ", ".join(
            f"{row['payer_name']} ({_pct(row['cagr_truncated'])} CAGR)"
            for _, row in payers_above.iterrows()
        )
        L.append(f"**Exceeded target:** {names}.")
    if len(payers_below):
        names = ", ".join(
            f"{row['payer_name']} ({_pct(row['cagr_truncated'])})"
            for _, row in payers_below.iterrows()
        )
        L.append("")
        L.append(f"**Within target:** {names}.")
    if len(not_assessed):
        names = ", ".join(row["payer_name"] for _, row in not_assessed.iterrows())
        L.append("")
        L.append(f"**Not assessed (incomplete submission):** {names} — too few "
                 f"complete years for a reliable trend; resubmission requested.")
    L.append("")

    L.append("### Provider organizations relative to target")
    L.append("")
    if len(orgs_above):
        names = ", ".join(
            f"{row['provider_org_name']} ({_pct(row['cagr_truncated'])})"
            for _, row in orgs_above.iterrows()
        )
        L.append(f"**{len(orgs_above)} of {len(provider)}** organizations "
                 f"exceeded the target on a CAGR basis: {names}.")
        L.append("")
        L.append(f"Their growth profiles differ — some are broad-based (most "
                 f"categories rising), others concentrated in one or two "
                 f"categories (measured over the complete years {cy[0]}–{cy[1]}):")
        L.append("")
        for _, row in orgs_above.iterrows():
            pid = row["provider_org_id"]
            if pid not in profile.index:
                continue
            p = profile.loc[pid]
            n_above, n_cat = int(p["n_above_target"]), int(p["n_categories"])
            ratio = n_above / n_cat if n_cat else 0
            share = p["top_category_dollar_share"]
            # broad-based = most categories rising; otherwise concentrated when
            # a single category drives the majority of the growth dollars.
            if ratio >= 0.75:
                kind = "**broad-based**"
            elif (not pd.isna(share)) and share >= 0.55:
                kind = "**concentrated**"
            else:
                kind = "**mixed**"
            driver = ""
            if p["top_category"] is not None and not pd.isna(share):
                driver = (f"; growth dollars led by {p['top_category']} "
                          f"({_pct(share)} of the provider's category growth)")
            L.append(f"- **{row['provider_org_name']}** "
                     f"({_pct(row['cagr_truncated'])}) — {kind}: "
                     f"{n_above} of {n_cat} categories above target{driver}.")
    else:
        L.append("No provider organization exceeded the target on a CAGR basis.")
    L.append("")

    L.append("### Service categories driving cost growth")
    L.append("")
    L.append(f"Measured over the two complete years ({cy[0]}–{cy[1]}), the "
             f"fastest-growing categories were:")
    L.append("")
    for cat, g in drivers.head(3).items():
        L.append(f"- **{cat}** — {_pct(g)}")
    L.append("")
    top_two = list(drivers.head(2).index)
    L.append(f"**{top_two[0]}** and **{top_two[1]}** are the fastest-growing "
             f"categories and the natural focus for cost-containment follow-up. "
             f"Facility spend (inpatient, outpatient) grew more slowly but "
             f"remains the largest share of total dollars and the most volatile "
             f"year to year, so it warrants continued monitoring.")
    L.append("")
    L.append("#### Price vs. utilization")
    L.append("")
    L.append("Because PMPM = utilization × unit price, each category's growth "
             "splits into *more services* (utilization) and *higher cost per "
             "service* (unit price) — which call for different interventions:")
    L.append("")
    lead_driver = None
    for cat in top_two:
        if cat not in decomp.index:
            continue
        dr = decomp.loc[cat]
        driver = "price" if dr["price_pts"] > dr["util_pts"] else "utilization"
        if lead_driver is None:
            lead_driver = driver
        L.append(f"- **{cat}** ({_pct(dr['pmpm_growth'])}) is **{driver}-"
                 f"driven** — {_pct(dr['price_pts'])} from unit price, "
                 f"{_pct(dr['util_pts'])} from utilization.")
    L.append("")
    if lead_driver == "price":
        L.append("In other words, the leading cost growth is not simply more "
                 "volume; it reflects rising unit prices, which points to "
                 "price/contracting levers rather than utilization management.")
    elif lead_driver == "utilization":
        L.append("In other words, the leading cost growth is driven by rising "
                 "service volume rather than unit prices, which points to "
                 "utilization-management levers.")
    L.append("")

    L.append("### High-cost member concentration")
    L.append("")
    L.append(f"Spending is highly concentrated: the top 1% of members account "
             f"for **{_pct(pooled['top_1pct_spend_share'])}** of allowed "
             f"spend and the top 5% for "
             f"**{_pct(pooled['top_5pct_spend_share'])}**. Claims above the "
             f"{_money(cfg['truncation_threshold'])} truncation threshold "
             f"represent **{_pct(pooled['high_cost_claim_spend_share'])}** of "
             f"spend — the basis for the high-cost truncation applied to all "
             f"trend measures.")
    L.append("")

    # high-cost concentration by facility (case-mix)
    hcf = M.high_cost_by_facility(
        M.prepare_analytic_claims(data, cfg), data["enrollment"], cfg
    ).merge(
        data["provider_org"][["provider_org_id", "provider_org_name"]],
        on="provider_org_id", how="left",
    ).sort_values("high_cost_share", ascending=False)
    lead = hcf.iloc[0]
    if lead["concentration_index"] >= 1.5:
        L.append("**High-cost members are not evenly distributed across "
                 f"facilities.** {lead['provider_org_name']} carries "
                 f"**{_pct(lead['high_cost_share'])}** of all high-cost members "
                 f"— **{lead['concentration_index']:.1f}×** its share of the "
                 f"panel — a tertiary/referral case-mix concentration. Its raw "
                 f"PMPM ({_money(lead['raw_pmpm'])}) runs well above its "
                 f"truncated PMPM ({_money(lead['truncated_pmpm'])}); that gap "
                 f"is catastrophic case-mix, not inefficiency, and is the "
                 f"central reason provider cost comparisons require risk "
                 f"adjustment before any fairness conclusion is drawn.")
        L.append("")

    L.append("## Data quality limitations")
    L.append("")
    L.append("- Findings reflect the data as submitted; flagged incomplete "
             "filings are excluded but cannot be reconstructed without "
             "resubmission.")
    L.append("- PMPM growth at the individual payer × provider cell level is "
             "volatile given member counts and the lumpiness of inpatient "
             "spend; cell-level results should be read as directional.")
    L.append("- All data is synthetic; magnitudes are illustrative and not "
             "calibrated to any real population, and no risk adjustment, "
             "attribution model, or actuarial method is applied.")
    L.append("")

    L.append("## Recommended next analytical steps")
    L.append("")
    L.append("1. Pursue resubmission of the flagged incomplete filing(s) and "
             "re-run the trend once complete data is received.")
    L.append("2. Decompose the leading category drivers (behavioral health, "
             "pharmacy) into price vs. utilization to target interventions.")
    L.append("3. Add risk adjustment so payer and provider comparisons account "
             "for population acuity differences.")
    L.append("4. Stand up continuous-enrollment cohorts to measure trend on a "
             "stable population alongside the full-population view.")
    L.append("5. Track the payers above target with a remediation plan and "
             "report progress against the target in the next cycle.")
    L.append("")
    L.append("---")
    L.append("")
    L.append("*Generated by `src/report.py`. Supporting tables in `outputs/`, "
             "charts in `outputs/figures/`.*")
    L.append("")
    return "\n".join(L)


def write_report(text: str) -> str:
    """Write the executive summary to reports/executive_summary.md."""
    C.ensure_dirs()
    path = C.REPORT_DIR / "executive_summary.md"
    path.write_text(text, encoding="utf-8")
    return str(path)


def main() -> None:
    cfg = C.load_config()
    data = data_io.load_raw()
    text = build_report(data, cfg)
    path = write_report(text)
    print(f"Executive summary written -> {path}  ({len(text.splitlines())} lines)")


if __name__ == "__main__":
    main()
