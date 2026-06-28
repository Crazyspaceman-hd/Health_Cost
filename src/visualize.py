"""Visualization layer.

Renders the program's headline charts to ``outputs/figures`` as PNGs suitable
for a slide deck or a data-quality / cost-growth dashboard:

    pmpm_by_year                PMPM trend (high-cost truncated)
    growth_vs_target            program YoY growth against the 3.4% target
    payer_cost_growth           per-payer cost growth (CAGR) vs target
    provider_cost_growth        per-provider-org cost growth (CAGR) vs target
    service_category_growth     category cost growth, complete years (drivers)
    utilization_per_1000        utilization per 1,000 member-months by category
    validation_by_severity      data-quality issue counts by severity
    payer_provider_heatmap      payer x provider cost-growth heatmap

Uses the non-interactive Agg backend so it runs headless (and in tests).
"""
from __future__ import annotations

from typing import Any

import matplotlib

matplotlib.use("Agg")  # headless; must precede pyplot import
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from . import config as C  # noqa: E402
from . import data_io  # noqa: E402
from . import metrics as M  # noqa: E402
from . import validate as V  # noqa: E402

# Severity -> color for consistent data-quality visuals.
SEVERITY_COLORS = {
    "CRITICAL": "#9e1b1b",
    "HIGH": "#d1495b",
    "MEDIUM": "#edae49",
    "LOW": "#66a182",
    "INFO": "#8d99ae",
}
_ACCENT = "#2e6f95"
# This is a COST-growth target: growth ABOVE target is the concern (red),
# growth at/BELOW target is on track (green). Lower is better.
_ABOVE = "#d1495b"   # above target = concern
_BELOW = "#66a182"   # below target = on track (good)


def _target_legend(target: float) -> list:
    """Legend handles that spell out the cost-growth orientation."""
    return [
        Patch(color=_ABOVE, label="Above target (concern)"),
        Patch(color=_BELOW, label="At / below target (on track)"),
        Line2D([0], [0], color="black", ls="--", label=f"Target {target:.1%}"),
    ]


def _style() -> None:
    plt.rcParams.update(
        {
            "figure.dpi": 120,
            "savefig.dpi": 120,
            "font.size": 10,
            "axes.titlesize": 12,
            "axes.titleweight": "bold",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.25,
            "grid.linestyle": "--",
        }
    )


def _save(fig: plt.Figure, name: str) -> str:
    C.ensure_dirs()
    path = C.FIG_DIR / f"{name}.png"
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return str(path)


# ---------------------------------------------------------------------------
# Individual charts
# ---------------------------------------------------------------------------
def fig_pmpm_by_year(exe: pd.DataFrame) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(exe["year"], exe["truncated_pmpm"], "-o", color=_ACCENT, lw=2.5)
    for _, r in exe.iterrows():
        ax.annotate(f"${r['truncated_pmpm']:,.0f}",
                    (r["year"], r["truncated_pmpm"]),
                    textcoords="offset points", xytext=(0, 8), ha="center")
    ax.set_title("Program PMPM by Year (high-cost truncated)")
    ax.set_xlabel("Measurement year")
    ax.set_ylabel("Allowed PMPM ($)")
    ax.set_xticks(exe["year"])
    return fig


def fig_growth_vs_target(exe: pd.DataFrame, target: float) -> plt.Figure:
    g = exe.dropna(subset=["yoy_growth"])
    fig, ax = plt.subplots(figsize=(7, 4))
    colors = [_ABOVE if v > target else _BELOW for v in g["yoy_growth"]]
    bars = ax.bar(g["year"].astype(str), g["yoy_growth"] * 100, color=colors)
    ax.axhline(target * 100, color="black", ls="--", lw=1.5)
    ax.bar_label(bars, fmt="%.1f%%", padding=3)
    ax.set_title("Year-over-Year Cost Growth vs Target (lower is better)")
    ax.set_ylabel("YoY truncated PMPM growth (%)")
    ax.set_xlabel("Measurement year")
    ax.legend(handles=_target_legend(target), loc="best", fontsize=8)
    return fig


_NOT_ASSESSED = "#bdbdbd"  # grey: trend not reliably assessable


def _growth_bars(
    df: pd.DataFrame, label_col: str, target: float, title: str,
    assessed_col: str | None = None,
) -> plt.Figure:
    d = df.drop_duplicates(label_col).copy()
    d = d.sort_values("cagr_truncated")
    # rows flagged as not assessable (e.g. an incomplete submission) are shown
    # in grey and NOT classified above/below target — consistent with how the
    # report treats them ("not assessed").
    assessed = (d[assessed_col] if assessed_col and assessed_col in d
                else pd.Series(True, index=d.index))
    labels = [f"{lab}  (not assessed)" if not ok else lab
              for lab, ok in zip(d[label_col].astype(str), assessed)]
    colors = [
        _NOT_ASSESSED if not ok else (_ABOVE if v > target else _BELOW)
        for v, ok in zip(d["cagr_truncated"], assessed)
    ]
    fig, ax = plt.subplots(figsize=(8, 4.5))
    bars = ax.barh(labels, d["cagr_truncated"] * 100, color=colors)
    ax.axvline(target * 100, color="black", ls="--", lw=1.5)
    ax.bar_label(bars, fmt="%.1f%%", padding=3)
    ax.set_title(title)
    ax.set_xlabel("Cost growth, CAGR (%)  —  lower is better")
    handles = _target_legend(target)
    if not assessed.all():
        handles.insert(2, Patch(color=_NOT_ASSESSED, label="Not assessed"))
    ax.legend(handles=handles, loc="lower right", fontsize=8)
    return fig


def fig_payer_cost_growth(payer: pd.DataFrame, target: float) -> plt.Figure:
    label = "payer_name" if "payer_name" in payer.columns else "payer_id"
    return _growth_bars(payer, label, target, "Cost Growth by Payer (CAGR vs Target)",
                        assessed_col="complete_submission")


def fig_provider_cost_growth(provider: pd.DataFrame, target: float) -> plt.Figure:
    label = "provider_org_name" if "provider_org_name" in provider.columns else "provider_org_id"
    return _growth_bars(
        provider, label, target, "Cost Growth by Provider Organization (CAGR vs Target)"
    )


def fig_service_category_growth(
    trends: pd.DataFrame, cfg: dict[str, Any]
) -> plt.Figure:
    """Category cost growth over the complete (non-incomplete) year pair."""
    years = sorted(cfg["years"])[:2]  # first two = complete years
    # 'Unclassified' is a data-quality bucket (claims with a missing service
    # category, flagged by DQ006), not a clinical category. Exclude it from the
    # cost-driver chart so its tiny, noisy trend doesn't masquerade as a driver;
    # it is quantified in the Data Quality view instead.
    trends = trends[trends["service_category"] != "Unclassified"]
    piv = trends.pivot(index="service_category", columns="year",
                       values="truncated_pmpm")
    target = float(cfg["target_growth_rate"])
    growth = (piv[years[1]] / piv[years[0]] - 1).sort_values() * 100
    fig, ax = plt.subplots(figsize=(8, 4.5))
    # color by the cost-growth orientation: categories growing faster than the
    # target (red) are the drivers/concern; slower (green) are contained.
    colors = [_ABOVE if g > target * 100 else _BELOW for g in growth.values]
    bars = ax.barh(growth.index, growth.values, color=colors)
    ax.bar_label(bars, fmt="%.1f%%", padding=3)
    ax.axvline(target * 100, color="black", ls="--", lw=1.5)
    ax.set_title(f"Service Category Cost Growth, {years[0]}–{years[1]} "
                 f"(complete years)")
    ax.set_xlabel("Truncated PMPM growth (%)  —  lower is better")
    ax.legend(handles=_target_legend(target), loc="lower right", fontsize=8)
    return fig


def fig_utilization_per_1000(trends: pd.DataFrame, cfg: dict[str, Any]) -> plt.Figure:
    year = sorted(cfg["years"])[1]  # latest complete year
    d = trends[(trends["year"] == year)
               & (trends["service_category"] != "Unclassified")]
    d = d.sort_values("utilization_per_1000_mm")
    fig, ax = plt.subplots(figsize=(8, 4.5))
    bars = ax.barh(d["service_category"], d["utilization_per_1000_mm"],
                   color=_ACCENT)
    ax.bar_label(bars, fmt="%.0f", padding=3)
    ax.set_title(f"Utilization per 1,000 Member-Months by Category ({year})")
    ax.set_xlabel("Claims per 1,000 member-months")
    return fig


def fig_validation_by_severity(summary: pd.DataFrame) -> plt.Figure:
    """Share of each check's population affected, colored by severity.

    Plots *proportions* of claim lines, not raw counts, so the magnitude reads
    honestly: even the most severe issues affect well under 1% of claims. All
    flagged records are excluded/de-duplicated before the cost analysis, so
    these are follow-up items, not analysis blockers.

    Restricted to **claim-level** checks so every bar shares one denominator
    (total claim lines) and is directly comparable; member- and submission-grain
    checks (enrollment gaps, completeness, incomplete submission) have their own
    denominators and appear in the validation table instead.
    """
    d = summary[(summary["issue_count"] > 0) & (summary["dimension"] == "claims")]
    d = d.sort_values("affected_pct")
    fig, ax = plt.subplots(figsize=(8, 4.8))
    colors = [SEVERITY_COLORS.get(s, "#888") for s in d["severity"]]
    bars = ax.barh(d["check_name"], d["affected_pct"], color=colors)
    ax.bar_label(bars, fmt="%.2f%%", padding=3, fontsize=8)
    ax.set_xlim(0, max(d["affected_pct"].max() * 1.3, 1.0))
    ax.set_title("Share of Claim Lines Affected, by Data-Quality Check")
    ax.set_xlabel("% of all claim lines (all flagged records are excluded "
                  "from analysis)")
    present = [s for s in V.SEVERITY_RANK if s in set(d["severity"])]
    ax.legend(
        handles=[Patch(color=SEVERITY_COLORS[s], label=s) for s in present],
        title="Severity", loc="lower right", framealpha=0.9,
    )
    return fig


def fig_provider_category_growth(
    data: dict[str, pd.DataFrame], cfg: dict[str, Any], top_n: int = 4,
    value: str = "growth",
) -> plt.Figure:
    """Grouped columns: service-category cost growth for the top-growth providers.

    Each provider group has one column per service category, so you can see
    which categories are driving each high-growth organization. Providers are
    the ``top_n`` with the highest overall truncated-PMPM CAGR.

    ``value`` selects what the columns measure:
      * ``"growth"``  — percent growth in truncated PMPM (rate);
      * ``"dollars"`` — the **PMPM dollar change** ($), which weights each
        category by size. A big-dollar category like Inpatient dominates here
        even at a modest growth rate, while a small category's large percentage
        barely registers — the complement the percentage view can't show.
    """
    target = float(cfg["target_growth_rate"])
    y0, y1 = sorted(cfg["years"])[:2]
    claims = M.prepare_analytic_claims(data, cfg)
    enroll = data["enrollment"]

    # rank providers by overall cost growth, keep the highest
    prov = M.cost_growth_by(
        claims[claims["provider_org_id"].notna()], enroll, "provider_org_id", cfg
    ).drop_duplicates("provider_org_id")
    top = prov.sort_values("cagr_truncated", ascending=False).head(top_n)
    top = top.merge(data["provider_org"][["provider_org_id", "provider_org_name"]],
                    on="provider_org_id", how="left")
    order = top["provider_org_id"].tolist()
    names = dict(zip(top["provider_org_id"], top["provider_org_name"]))

    pc = M.provider_category_growth(claims, enroll, cfg)
    pc = pc[pc["provider_org_id"].isin(order)
            & (pc["service_category"] != "Unclassified")].copy()

    dollars = value == "dollars"
    if dollars:
        pc["value"] = pc["pmpm_end"] - pc["pmpm_start"]   # $ PMPM change
    else:
        pc["value"] = pc["growth"] * 100                  # % growth

    # category order/colors from config (stable, excludes Unclassified)
    cats = [s["service_category"] for s in cfg["service_categories"]]
    grid = pc.pivot(index="provider_org_id", columns="service_category",
                    values="value").reindex(index=order, columns=cats)

    n_cat = len(cats)
    width = 0.8 / n_cat
    x = np.arange(len(order))
    palette = plt.get_cmap("tab10")(np.linspace(0, 1, 10))
    fig, ax = plt.subplots(figsize=(11, 5.5))
    for i, cat in enumerate(cats):
        offs = (i - (n_cat - 1) / 2) * width
        ax.bar(x + offs, grid[cat].to_numpy(), width, label=cat, color=palette[i])
    if not dollars:
        ax.axhline(target * 100, color="black", ls="--", lw=1.5,
                   label=f"Target {target:.1%}")
    ax.axhline(0, color="#888", lw=0.8)
    ax.set_xticks(x, [names.get(p, p) for p in order])
    unit = "PMPM Dollar Change ($)" if dollars else "Cost Growth (%)"
    ax.set_title(f"{unit.split(' (')[0]} by Service Category — "
                 f"Highest-Growth Providers ({y0}–{y1})")
    ax.set_ylabel(f"{'$ PMPM change' if dollars else 'Truncated PMPM growth (%)'}"
                  f"  —  lower is better")
    ax.legend(ncol=3, fontsize=8, loc="upper center",
              bbox_to_anchor=(0.5, -0.12))
    return fig


def fig_price_utilization(
    data: dict[str, pd.DataFrame], cfg: dict[str, Any]
) -> plt.Figure:
    """Per-category cost growth split into utilization vs unit-price components.

    Two bars per category (utilization growth, unit-price growth) with the total
    PMPM growth marked, so it's clear whether a category's growth comes from
    more services or a higher cost per service.
    """
    claims = M.prepare_analytic_claims(data, cfg)
    dec = M.price_utilization_decomposition(claims, data["enrollment"], cfg)
    dec = dec[dec["service_category"] != "Unclassified"].sort_values("pmpm_growth")

    cats = dec["service_category"].tolist()
    y = np.arange(len(cats))
    h = 0.38
    fig, ax = plt.subplots(figsize=(8.5, 5.5))
    ax.barh(y + h / 2, dec["util_growth"] * 100, h, color="#4c9f70",
            label="Utilization (more services)")
    ax.barh(y - h / 2, dec["price_growth"] * 100, h, color="#c9622e",
            label="Unit price ($ per service)")
    ax.plot(dec["pmpm_growth"] * 100, y, "D", color="black", ms=6,
            label="Total PMPM growth")
    ax.axvline(0, color="#888", lw=0.8)
    ax.set_yticks(y, cats)
    ax.set_title("Cost Growth: Price vs. Utilization by Service Category")
    ax.set_xlabel("Growth, complete years (%)")
    ax.legend(loc="lower right", fontsize=8, framealpha=0.9)
    return fig


def fig_validation_resolution(summary: pd.DataFrame) -> plt.Figure:
    """Found vs. resolved-before-analysis, per claim-level check.

    For each claim-level check, the full bar is the records *found* and the
    overlaid green bar is how many are *resolved* (de-duplicated, excluded, or
    truncated) before any cost number is computed. Checks where the records are
    retained (e.g. missing provider ID) show no green fill — a visible reminder
    that they still need submitter follow-up.
    """
    d = summary[(summary["issue_count"] > 0) & (summary["dimension"] == "claims")]
    d = d.sort_values("issue_count")
    y = np.arange(len(d))
    fig, ax = plt.subplots(figsize=(8, 4.8))
    ax.barh(y, d["issue_count"], color="#d9d9d9", label="Found")
    ax.barh(y, d["resolved_count"], color="#2a9d8f", label="Resolved before analysis")
    ax.set_yticks(y, d["check_name"])
    for i, (_, r) in enumerate(d.iterrows()):
        ax.text(r["issue_count"], i, f"  {int(r['resolved_count']):,}/"
                f"{int(r['issue_count']):,}", va="center", fontsize=8)
    total = int(d["issue_count"].sum())
    resolved = int(d["resolved_count"].sum())
    ax.set_title(f"Data-Quality Issues: Found vs Resolved Before Analysis "
                 f"({resolved:,}/{total:,} handled)")
    ax.set_xlabel("Claim records")
    ax.legend(loc="lower right", framealpha=0.9)
    return fig


def fig_high_cost_concentration(
    data: dict[str, pd.DataFrame], cfg: dict[str, Any]
) -> plt.Figure:
    """Lorenz curve of allowed spend across members, with the Gini coefficient.

    Visualizes how concentrated spending is: members are ranked from lowest to
    highest total allowed spend, and the curve plots cumulative share of spend
    against cumulative share of members. The further the curve bows below the
    line of equality, the more concentrated the spend.
    """
    claims = M.prepare_analytic_claims(data, cfg)
    spend = (
        claims.groupby("member_id")["allowed_amount"].sum()
        .clip(lower=0).sort_values().to_numpy()
    )
    n = len(spend)
    total = spend.sum()
    cum_spend = np.insert(np.cumsum(spend) / total, 0, 0.0)
    cum_members = np.insert(np.arange(1, n + 1) / n, 0, 0.0)
    gini = 1 - 2 * np.trapezoid(cum_spend, cum_members)

    # share of spend held by the top 1% / 5% / 10% of members
    def top_share(p: float) -> float:
        k = max(1, int(n * p))
        return spend[-k:].sum() / total

    fig, ax = plt.subplots(figsize=(7.5, 5.5))
    ax.plot([0, 1], [0, 1], "--", color="#888", lw=1.3, label="Line of equality")
    ax.plot(cum_members, cum_spend, color=_ACCENT, lw=2.5, label="Lorenz curve")
    ax.fill_between(cum_members, cum_spend, cum_members, color=_ACCENT, alpha=0.12)

    # annotate the top-5% reference point
    x5 = 0.95
    y5 = np.interp(x5, cum_members, cum_spend)
    ax.axvline(x5, color=_ABOVE, ls=":", lw=1.2)
    ax.annotate(
        f"Top 5% of members\n→ {top_share(0.05):.0%} of spend",
        xy=(x5, y5), xytext=(0.45, 0.30),
        arrowprops=dict(arrowstyle="->", color=_ABOVE), color="#333", fontsize=9,
    )
    ax.set_title(f"High-Cost Member Concentration  (Gini = {gini:.2f})")
    ax.set_xlabel("Cumulative share of members (lowest → highest spend)")
    ax.set_ylabel("Cumulative share of allowed spend")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend(loc="upper left")
    ax.text(0.02, 0.60,
            f"Top 1%: {top_share(0.01):.0%}\nTop 10%: {top_share(0.10):.0%}",
            fontsize=9, color="#333",
            bbox=dict(boxstyle="round", fc="white", ec="#ccc", alpha=0.9))
    return fig


def fig_high_cost_by_facility(
    data: dict[str, pd.DataFrame], cfg: dict[str, Any]
) -> plt.Figure:
    """Share of high-cost members by facility vs. each facility's panel share.

    Bars show what share of all high-cost members each facility holds; the
    diamonds mark the share expected if high-cost members were spread in
    proportion to panel size. A facility well above its diamond is a high
    case-mix concentration (a tertiary/referral pattern).
    """
    claims = M.prepare_analytic_claims(data, cfg)
    hcf = M.high_cost_by_facility(claims, data["enrollment"], cfg).merge(
        data["provider_org"][["provider_org_id", "provider_org_name"]],
        on="provider_org_id", how="left",
    ).sort_values("high_cost_share")

    y = np.arange(len(hcf))
    # flag over-concentrated facilities (well above proportional)
    colors = [_ABOVE if ix >= 1.5 else _ACCENT
              for ix in hcf["concentration_index"]]
    fig, ax = plt.subplots(figsize=(8.5, 5))
    bars = ax.barh(y, hcf["high_cost_share"] * 100, color=colors,
                   label="Share of high-cost members")
    ax.plot(hcf["panel_share"] * 100, y, "D", color="black", ms=6,
            label="Expected if proportional (panel share)")
    ax.bar_label(bars, labels=[f" {v:.0f}%  ({ix:.1f}×)" for v, ix in
                               zip(hcf["high_cost_share"] * 100,
                                   hcf["concentration_index"])],
                 fontsize=8)
    ax.set_yticks(y, hcf["provider_org_name"])
    ax.set_xlabel("Share of all high-cost members (%)   ·   (×) = vs. panel share")
    ax.set_title("High-Cost Member Concentration by Facility")
    ax.legend(loc="lower right", fontsize=8, framealpha=0.9)
    ax.margins(x=0.18)
    return fig


def fig_payer_provider_heatmap(
    data: dict[str, pd.DataFrame], cfg: dict[str, Any]
) -> plt.Figure:
    """Heatmap of truncated-PMPM CAGR for each payer x provider cell."""
    trunc = float(cfg["truncation_threshold"])
    target = float(cfg["target_growth_rate"])
    claims = M.prepare_analytic_claims(data, cfg)

    spend = M._spend(claims, ["payer_id", "provider_org_id", "year"], trunc)
    mm = M.member_months(
        data["enrollment"], ["payer_id", "provider_org_id", "year"]
    )
    df = spend.join(mm, how="left").reset_index()
    df = df[df["member_months"] > 0]
    df["pmpm"] = df["total_allowed_trunc"] / df["member_months"]
    cagr = (
        df.set_index("year").groupby(["payer_id", "provider_org_id"])["pmpm"]
        .apply(M._annualized_growth).rename("cagr").reset_index()
    )
    grid = cagr.pivot(index="payer_id", columns="provider_org_id",
                      values="cagr") * 100

    # map IDs to readable payer / provider names for the axis labels
    payer_names = dict(zip(data["payer"]["payer_id"],
                           data["payer"]["payer_name"]))
    org_names = dict(zip(data["provider_org"]["provider_org_id"],
                         data["provider_org"]["provider_org_name"]))
    xlabels = [org_names.get(c, c) for c in grid.columns]
    ylabels = [payer_names.get(r, r) for r in grid.index]

    fig, ax = plt.subplots(figsize=(10, 5.5))
    lim = np.nanmax(np.abs(grid.values - target * 100))
    im = ax.imshow(grid.values, cmap="RdYlGn_r",
                   vmin=target * 100 - lim, vmax=target * 100 + lim)
    ax.set_xticks(range(len(grid.columns)), xlabels, rotation=35,
                  ha="right", fontsize=8)
    ax.set_yticks(range(len(grid.index)), ylabels, fontsize=8)
    for i in range(grid.shape[0]):
        for j in range(grid.shape[1]):
            v = grid.values[i, j]
            if not np.isnan(v):
                ax.text(j, i, f"{v:.1f}", ha="center", va="center", fontsize=8)
    fig.colorbar(im, ax=ax, label="Cost growth CAGR (%)")
    ax.set_title(f"Payer × Provider Cost Growth (CAGR %, target {target:.1%})")
    ax.set_xlabel("Provider organization")
    ax.set_ylabel("Payer")
    return fig


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------
def build_all_figures(
    data: dict[str, pd.DataFrame], cfg: dict[str, Any] | None = None
) -> dict[str, str]:
    """Render every figure to outputs/figures and return name -> path."""
    if cfg is None:
        cfg = C.load_config()
    _style()
    target = float(cfg["target_growth_rate"])

    # Cost-growth charts use the completeness-adjusted views (incomplete
    # submissions excluded); the data-quality chart reflects ALL raw issues.
    cells = M.incomplete_payer_years(data, cfg)
    tables = M.build_metric_tables(data, cfg, exclude_incomplete=True)
    # heatmap: drop the incomplete payer entirely for a consistent panel
    pooled = M.drop_payers(data, {p for p, _ in cells})
    summary = V.build_summary(V.run_validation(data, cfg))
    note = (
        "Excludes incomplete submission(s): "
        + ", ".join(f"{p} {y}" for p, y in sorted(cells))
        if cells else ""
    )

    figures = {
        "pmpm_by_year": fig_pmpm_by_year(tables["executive_summary_metrics"]),
        "growth_vs_target": fig_growth_vs_target(
            tables["executive_summary_metrics"], target),
        "payer_cost_growth": fig_payer_cost_growth(
            tables["payer_cost_growth_summary"], target),
        "provider_cost_growth": fig_provider_cost_growth(
            tables["provider_cost_growth_summary"], target),
        "provider_category_growth": fig_provider_category_growth(pooled, cfg),
        "provider_category_dollar_growth": fig_provider_category_growth(
            pooled, cfg, value="dollars"),
        "service_category_growth": fig_service_category_growth(
            tables["service_category_trends"], cfg),
        "utilization_per_1000": fig_utilization_per_1000(
            tables["service_category_trends"], cfg),
        "price_vs_utilization": fig_price_utilization(pooled, cfg),
        "validation_by_severity": fig_validation_by_severity(summary),
        "validation_resolution": fig_validation_resolution(summary),
        "high_cost_concentration": fig_high_cost_concentration(data, cfg),
        "high_cost_by_facility": fig_high_cost_by_facility(data, cfg),
        "payer_provider_heatmap": fig_payer_provider_heatmap(pooled, cfg),
    }
    if note:
        for key in ("payer_cost_growth", "provider_cost_growth",
                    "growth_vs_target", "pmpm_by_year"):
            # place below the whole figure, right-aligned, so it never collides
            # with axis labels or bar values
            figures[key].text(0.99, -0.04, note, fontsize=7.5, style="italic",
                              color="#666", ha="right", va="top",
                              transform=figures[key].axes[0].transAxes)
    return {name: _save(fig, name) for name, fig in figures.items()}


def main() -> None:
    cfg = C.load_config()
    data = data_io.load_raw()
    written = build_all_figures(data, cfg)
    print(f"Rendered {len(written)} figures to {C.FIG_DIR}:\n")
    for name, path in written.items():
        print(f"  {name:26s} -> {path}")


if __name__ == "__main__":
    main()
