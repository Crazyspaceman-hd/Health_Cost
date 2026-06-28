"""Data-quality validation engine.

Runs a battery of validation checks against the synthetic source tables and
emits two report-friendly artifacts:

    outputs/validation_summary.csv  -- one row per check: severity, issue count,
                                       affected %, description, recommended action
    outputs/data_quality_issues.csv -- one row per affected record: the check,
                                       severity, the offending entity, and detail

Each check is an independent function registered in ``CHECKS``. A check receives
the full table dict and returns a :class:`CheckResult`. This mirrors how a real
data-submission program runs a documented rule set over each annual filing.

Severity scale (most to least urgent):
    CRITICAL  financial / identity integrity broken (negatives, dup IDs, paid>allowed)
    HIGH      records that cannot be trusted or attributed (orphans, bad dates,
              incomplete submissions)
    MEDIUM    completeness gaps that bias analysis (missing fields, enrollment gaps)
    LOW       informational / expected-but-worth-review (high-cost outliers)
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np
import pandas as pd

from . import config as C
from . import data_io

# Severity ordering for sorting and status derivation.
SEVERITY_RANK = {"CRITICAL": 0, "HIGH": 1, "MEDIUM": 2, "LOW": 3, "INFO": 4}

# How the analytic pipeline handles each check's records, and whether that
# handling NEUTRALIZES their impact on the cost analysis (the "resolved" flag).
# Used to show that detected issues are actually dealt with before measurement,
# not just reported.
HANDLING = {
    "DQ001": ("De-duplicated (keep first)", True),
    "DQ002": ("Excluded from cost metrics", True),
    "DQ003": ("Excluded from cost metrics", True),
    "DQ004": ("Would be rejected at intake", False),
    "DQ005": ("Retained; dropped from provider cuts", False),
    "DQ006": ("Relabeled 'Unclassified'; off category charts", True),
    "DQ007": ("Excluded from cost metrics", True),
    "DQ008": ("Excluded from cost metrics", True),
    "DQ009": ("Excluded from cost metrics", True),
    "DQ010": ("Flagged for review (expected churn)", False),
    "DQ011": ("Truncated at threshold for trend", True),
    "DQ012": ("Flagged for submitter follow-up", False),
    "DQ013": ("Held out of trend results", True),
}

# Standard columns for the per-record detail table.
DETAIL_COLS = ["entity_type", "entity_id", "detail"]


@dataclass
class CheckResult:
    """Outcome of a single validation check."""

    check_id: str
    check_name: str
    dimension: str          # which table / grain the check applies to
    severity: str
    issue_count: int
    denominator: int        # population the rate is measured against
    description: str
    recommendation: str
    details: pd.DataFrame = field(
        default_factory=lambda: pd.DataFrame(columns=DETAIL_COLS)
    )

    @property
    def affected_pct(self) -> float:
        return 0.0 if self.denominator == 0 else self.issue_count / self.denominator

    @property
    def status(self) -> str:
        """PASS when nothing found; REVIEW for informational; FAIL otherwise."""
        if self.issue_count == 0:
            return "PASS"
        return "REVIEW" if self.severity in ("LOW", "INFO") else "FAIL"


def _detail(entity_type: str, ids, details) -> pd.DataFrame:
    """Build a standardized detail frame from arrays/series."""
    return pd.DataFrame(
        {
            "entity_type": entity_type,
            "entity_id": np.asarray(ids, dtype=object),
            "detail": np.asarray(details, dtype=object),
        }
    )


# ---------------------------------------------------------------------------
# Individual checks
# ---------------------------------------------------------------------------
def check_duplicate_claim_ids(data, cfg) -> CheckResult:
    c = data["claims"]
    dup_mask = c["claim_id"].duplicated(keep=False)
    dups = c.loc[dup_mask, "claim_id"]
    # count = redundant rows (every occurrence beyond the first)
    redundant = int(c["claim_id"].duplicated(keep="first").sum())
    sample_ids = dups.drop_duplicates()
    details = _detail(
        "claim", sample_ids,
        [f"claim_id appears {n}x" for n in dups.value_counts().loc[sample_ids]],
    )
    return CheckResult(
        "DQ001", "Duplicate claim IDs", "claims", "CRITICAL",
        redundant, len(c),
        "Claim IDs that appear on more than one row (exact resubmissions).",
        "De-duplicate on claim_id before financial aggregation; confirm with "
        "submitter whether these are true resubmissions or load errors.",
        details,
    )


def check_negative_amounts(data, cfg) -> CheckResult:
    c = data["claims"]
    mask = (c["allowed_amount"] < 0) | (c["paid_amount"] < 0)
    bad = c.loc[mask]
    details = _detail(
        "claim", bad["claim_id"],
        "allowed=" + bad["allowed_amount"].round(2).astype(str)
        + " paid=" + bad["paid_amount"].round(2).astype(str),
    )
    return CheckResult(
        "DQ002", "Negative allowed/paid amounts", "claims", "CRITICAL",
        int(mask.sum()), len(c),
        "Claims with a negative allowed or paid amount.",
        "Investigate as reversals/adjustments; exclude or net against the "
        "original claim before computing total medical expense.",
        details,
    )


def check_paid_gt_allowed(data, cfg) -> CheckResult:
    c = data["claims"]
    # only meaningful where both amounts are present and allowed is non-negative
    mask = (
        c["paid_amount"].notna()
        & c["allowed_amount"].notna()
        & (c["allowed_amount"] >= 0)
        & (c["paid_amount"] > c["allowed_amount"] + 0.01)
    )
    bad = c.loc[mask]
    details = _detail(
        "claim", bad["claim_id"],
        "paid=" + bad["paid_amount"].round(2).astype(str)
        + " > allowed=" + bad["allowed_amount"].round(2).astype(str),
    )
    return CheckResult(
        "DQ003", "Paid amount exceeds allowed amount", "claims", "CRITICAL",
        int(mask.sum()), len(c),
        "Paid amount is greater than the allowed amount, which is impossible "
        "under standard adjudication.",
        "Return to submitter for correction; do not include in paid-amount "
        "trend until resolved.",
        details,
    )


def check_missing_payer_id(data, cfg) -> CheckResult:
    c = data["claims"]
    mask = c["payer_id"].isna()
    details = _detail("claim", c.loc[mask, "claim_id"], "missing payer_id")
    return CheckResult(
        "DQ004", "Missing payer ID", "claims", "HIGH",
        int(mask.sum()), len(c),
        "Claims with no payer identifier.",
        "Cannot attribute spend to a submitter; reject and request resubmission.",
        details,
    )


def check_missing_provider_id(data, cfg) -> CheckResult:
    c = data["claims"]
    mask = c["provider_org_id"].isna()
    details = _detail("claim", c.loc[mask, "claim_id"], "missing provider_org_id")
    return CheckResult(
        "DQ005", "Missing provider organization ID", "claims", "MEDIUM",
        int(mask.sum()), len(c),
        "Claims with no attributed provider organization.",
        "Excluded from provider-level cost growth; follow up on attribution logic.",
        details,
    )


def check_missing_service_category(data, cfg) -> CheckResult:
    c = data["claims"]
    mask = c["service_category"].isna()
    details = _detail("claim", c.loc[mask, "claim_id"], "missing service_category")
    return CheckResult(
        "DQ006", "Missing service category", "claims", "MEDIUM",
        int(mask.sum()), len(c),
        "Claims with no service category, which cannot be assigned to a "
        "category trend.",
        "Map to a category from procedure detail where possible; otherwise "
        "report as 'Unclassified' and quantify the unmapped share.",
        details,
    )


def check_invalid_dates(data, cfg) -> CheckResult:
    c = data["claims"].copy()
    years = sorted(cfg["years"])
    lo = pd.Timestamp(f"{years[0]}-01-01")
    hi = pd.Timestamp(f"{years[-1]}-12-31")
    sd = pd.to_datetime(c["service_date"], errors="coerce")
    rd = pd.to_datetime(c["received_date"], errors="coerce")
    bad_service = sd.isna() | (sd < lo) | (sd > hi)
    received_before_service = rd.notna() & sd.notna() & (rd < sd)
    mask = bad_service | received_before_service
    bad = c.loc[mask]
    reason = np.where(
        bad_service.loc[mask],
        "service_date out of range/null",
        "received_date precedes service_date",
    )
    details = _detail("claim", bad["claim_id"], reason)
    return CheckResult(
        "DQ007", "Invalid service/received dates", "claims", "HIGH",
        int(mask.sum()), len(c),
        f"Service dates outside the measurement window "
        f"({years[0]}-{years[-1]}) or receipt dates before service.",
        "Quarantine affected claims; confirm date fields with submitter.",
        details,
    )


def check_orphan_claims(data, cfg) -> CheckResult:
    """Claims for members that never appear in enrollment."""
    c = data["claims"]
    enrolled = set(data["enrollment"]["member_id"].unique())
    mask = ~c["member_id"].isin(enrolled)
    bad = c.loc[mask]
    details = _detail(
        "claim", bad["claim_id"],
        "member " + bad["member_id"].astype(str) + " has no enrollment",
    )
    return CheckResult(
        "DQ008", "Claims with no matching enrollment (orphans)", "claims", "HIGH",
        int(mask.sum()), len(c),
        "Claims whose member_id is absent from the enrollment file.",
        "Member-months are missing for these members; PMPM denominators are "
        "understated. Request the missing enrollment records.",
        details,
    )


def check_claims_outside_enrollment(data, cfg) -> CheckResult:
    """Claims for enrolled members but in a month they were not covered."""
    c = data["claims"].copy()
    enr = data["enrollment"]
    enrolled_members = set(enr["member_id"].unique())
    has_coverage = c["member_id"].isin(enrolled_members)

    claim_month = pd.to_datetime(c["service_date"], errors="coerce").dt.to_period(
        "M"
    ).astype(str)
    enr_keys = pd.MultiIndex.from_frame(
        enr[["member_id", "enrollment_month"]].astype(str)
    )
    claim_keys = pd.MultiIndex.from_arrays(
        [c["member_id"].astype(str), claim_month]
    )
    in_enrollment = claim_keys.isin(enr_keys)
    # covered member, valid month, but not an enrolled month for that member
    mask = has_coverage & ~in_enrollment & claim_month.ne("NaT")
    bad = c.loc[mask]
    details = _detail(
        "claim", bad["claim_id"],
        "service month " + claim_month.loc[mask] + " outside member coverage",
    )
    return CheckResult(
        "DQ009", "Claims outside enrollment period", "claims", "HIGH",
        int(mask.sum()), len(c),
        "Claims dated in a month the member was not enrolled (coverage gap or "
        "bad service date).",
        "Reconcile against eligibility; these distort utilization and PMPM.",
        details,
    )


def check_enrollment_gaps(data, cfg) -> CheckResult:
    """Members whose enrolled months are non-contiguous."""
    enr = data["enrollment"].copy()
    # month ordinals: consecutive months differ by exactly 1, so a contiguous
    # run satisfies (max - min + 1) == count. Anything larger implies a gap.
    enr["ord"] = pd.PeriodIndex(enr["enrollment_month"], freq="M").asi8
    grp = enr.groupby("member_id")["ord"]
    span = (grp.max() - grp.min()) + 1
    months = grp.count()
    gaps = months[span > months]
    details = _detail(
        "member", gaps.index,
        "enrolled " + months.loc[gaps.index].astype(str) + " of "
        + span.loc[gaps.index].astype(str) + " spanned months",
    )
    return CheckResult(
        "DQ010", "Enrollment gaps (non-contiguous coverage)", "enrollment",
        "MEDIUM", int(len(gaps)), int(enr["member_id"].nunique()),
        "Members with a break in otherwise continuous enrollment.",
        "Expected for real churn, but verify these are true disenrollments and "
        "not dropped eligibility records; affects continuous-enrollment cohorts.",
        details,
    )


def check_high_cost_outliers(data, cfg) -> CheckResult:
    """Unusually high-cost claims (informational; drives truncation)."""
    c = data["claims"]
    thresh = float(cfg["truncation_threshold"]) * float(
        cfg["validation"]["high_cost_review_multiple"]
    )
    mask = c["allowed_amount"] > thresh
    bad = c.loc[mask]
    details = _detail(
        "claim", bad["claim_id"],
        "allowed=" + bad["allowed_amount"].round(0).astype(int).astype(str),
    )
    return CheckResult(
        "DQ011", "High-cost outlier claims", "claims", "LOW",
        int(mask.sum()), len(c),
        f"Claims with allowed amount above the ${thresh:,.0f} truncation "
        f"threshold.",
        "Expected catastrophic claims; review for coding errors, then apply "
        "high-cost truncation before measuring underlying trend.",
        details,
    )


def check_payer_provider_completeness(data, cfg) -> CheckResult:
    """Payer/provider/year cells with members but anomalously few claims."""
    v = cfg["validation"]
    c = data["claims"].copy()
    c["yr"] = pd.to_datetime(c["service_date"], errors="coerce").dt.year
    years = set(cfg["years"])
    c = c[c["yr"].isin(years) & c["provider_org_id"].notna()]

    claims_by = c.groupby(["payer_id", "provider_org_id", "yr"]).size()
    mm_by = (
        data["enrollment"]
        .groupby(["payer_id", "provider_org_id", "year"])
        .size()
        .rename_axis(["payer_id", "provider_org_id", "yr"])
    )
    cell = pd.concat(
        [claims_by.rename("claims"), mm_by.rename("mm")], axis=1
    ).fillna(0)
    cell["cpm"] = np.where(cell["mm"] > 0, cell["claims"] / cell["mm"], np.nan)

    # each cell's "norm" = median cpm of the same payer/provider across years
    norm = cell.groupby(level=["payer_id", "provider_org_id"])["cpm"].transform(
        "median"
    )
    floor = float(v["min_member_months_for_completeness"])
    ratio = float(v["completeness_ratio_threshold"])
    mask = (cell["mm"] >= floor) & (cell["cpm"] < ratio * norm)
    flagged = cell.loc[mask].reset_index()
    details = _detail(
        "payer-provider-year",
        flagged["payer_id"] + "|" + flagged["provider_org_id"]
        + "|" + flagged["yr"].astype(str),
        "claims/MM " + flagged["cpm"].round(3).astype(str)
        + " vs norm " + norm.loc[mask].round(3).values.astype(str),
    )
    return CheckResult(
        "DQ012", "Payer/provider cells with suspiciously missing data",
        "claims+enrollment", "HIGH", int(mask.sum()),
        int(len(cell)),
        "Payer/provider/year cells whose claims-per-member-month fall well "
        "below that submitter's own norm — a likely partial submission.",
        "Confirm completeness of the filing for these cells before including "
        "them in cost-growth results.",
        details,
    )


def check_yoy_volume_change(data, cfg) -> CheckResult:
    """Year-over-year payer volume swings suggesting incomplete submissions."""
    v = cfg["validation"]
    c = data["claims"].copy()
    c["yr"] = pd.to_datetime(c["service_date"], errors="coerce").dt.year
    years = sorted(cfg["years"])
    c = c[c["yr"].isin(set(years))]

    vol = c.groupby(["payer_id", "yr"]).size().rename("claims")
    mm = (
        data["enrollment"].groupby(["payer_id", "year"]).size()
        .rename_axis(["payer_id", "yr"]).rename("mm")
    )
    df = pd.concat([vol, mm], axis=1).fillna(0)
    df["cpm"] = np.where(df["mm"] > 0, df["claims"] / df["mm"], np.nan)
    df = df.reset_index().sort_values(["payer_id", "yr"])
    df["prev_cpm"] = df.groupby("payer_id")["cpm"].shift(1)
    df["yoy"] = df["cpm"] / df["prev_cpm"] - 1

    drop = -float(v["yoy_volume_drop_threshold"])
    spike = float(v["yoy_volume_spike_threshold"])
    mask = df["yoy"].notna() & ((df["yoy"] < drop) | (df["yoy"] > spike))
    flagged = df.loc[mask]
    details = _detail(
        "payer-year",
        flagged["payer_id"] + "|" + flagged["yr"].astype(str),
        "claims/MM YoY " + (flagged["yoy"] * 100).round(1).astype(str) + "%",
    )
    return CheckResult(
        "DQ013", "Year-over-year volume change (possible incomplete submission)",
        "claims+enrollment", "HIGH", int(mask.sum()),
        int(df["yoy"].notna().sum()),
        "Payer-year claims-per-member-month that moved more than the allowed "
        "threshold versus the prior year.",
        "A large drop usually signals an incomplete annual submission; exclude "
        "or footnote the affected payer-year in trend reporting.",
        details,
    )


# Ordered registry of all checks the engine runs.
CHECKS: list[Callable[[dict, dict], CheckResult]] = [
    check_duplicate_claim_ids,
    check_negative_amounts,
    check_paid_gt_allowed,
    check_missing_payer_id,
    check_missing_provider_id,
    check_missing_service_category,
    check_invalid_dates,
    check_orphan_claims,
    check_claims_outside_enrollment,
    check_enrollment_gaps,
    check_high_cost_outliers,
    check_payer_provider_completeness,
    check_yoy_volume_change,
]


# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------
def run_validation(
    data: dict[str, pd.DataFrame], cfg: dict[str, Any] | None = None
) -> list[CheckResult]:
    """Run every registered check and return the results."""
    if cfg is None:
        cfg = C.load_config()
    return [check(data, cfg) for check in CHECKS]


def build_summary(results: list[CheckResult]) -> pd.DataFrame:
    """Collapse results into the report-friendly validation_summary table."""
    rows = []
    for r in results:
        handling, neutralized = HANDLING.get(r.check_id, ("Flagged for review", False))
        rows.append({
            "check_id": r.check_id,
            "check_name": r.check_name,
            "dimension": r.dimension,
            "severity": r.severity,
            "status": r.status,
            "issue_count": r.issue_count,
            "denominator": r.denominator,
            "affected_pct": round(r.affected_pct * 100, 4),
            "handling": handling,
            # records "resolved" before analysis = handled in a way that
            # neutralizes their impact on the cost metrics
            "resolved_count": r.issue_count if neutralized else 0,
            "description": r.description,
            "recommended_follow_up": r.recommendation,
        })
    df = pd.DataFrame(rows)
    df["_rank"] = df["severity"].map(SEVERITY_RANK)
    df = df.sort_values(["_rank", "issue_count"], ascending=[True, False])
    return df.drop(columns="_rank").reset_index(drop=True)


def build_issue_detail(results: list[CheckResult], max_per_check: int = 1000) -> pd.DataFrame:
    """Stack per-record detail rows from every check into one long table.

    Capped per check so the file stays report-sized; the true count always
    lives in the summary table.
    """
    frames = []
    for r in results:
        if r.details.empty:
            continue
        d = r.details.head(max_per_check).copy()
        d.insert(0, "check_id", r.check_id)
        d.insert(1, "check_name", r.check_name)
        d.insert(2, "severity", r.severity)
        frames.append(d)
    if not frames:
        return pd.DataFrame(
            columns=["check_id", "check_name", "severity", *DETAIL_COLS]
        )
    return pd.concat(frames, ignore_index=True)


def write_outputs(
    summary: pd.DataFrame, detail: pd.DataFrame
) -> dict[str, str]:
    """Write the two validation artifacts to the outputs directory."""
    C.ensure_dirs()
    spath = C.OUTPUT_DIR / "validation_summary.csv"
    dpath = C.OUTPUT_DIR / "data_quality_issues.csv"
    summary.to_csv(spath, index=False)
    detail.to_csv(dpath, index=False)
    return {"summary": str(spath), "detail": str(dpath)}


def main() -> None:
    cfg = C.load_config()
    data = data_io.load_raw()
    results = run_validation(data, cfg)
    summary = build_summary(results)
    detail = build_issue_detail(results)
    paths = write_outputs(summary, detail)

    n_fail = (summary["status"] == "FAIL").sum()
    n_review = (summary["status"] == "REVIEW").sum()
    print("Validation complete.\n")
    cols = ["check_id", "severity", "status", "issue_count", "check_name"]
    with pd.option_context("display.max_colwidth", 48, "display.width", 120):
        print(summary[cols].to_string(index=False))
    print(
        f"\n  {n_fail} check(s) FAILED, {n_review} flagged for REVIEW, "
        f"{len(summary) - n_fail - n_review} PASSED."
    )
    print(f"  Summary -> {paths['summary']}")
    print(f"  Detail  -> {paths['detail']}  ({len(detail):,} record(s))")


if __name__ == "__main__":
    main()
