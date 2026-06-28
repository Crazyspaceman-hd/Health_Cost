"""Streamlit dashboard for the Health Cost Growth Target analytics.

An interactive front-end over the same engines used by the batch pipeline — it
reuses ``metrics``, ``validate`` and ``visualize`` rather than recomputing
anything, so the dashboard and the generated report always agree.

Run with:

    streamlit run src/dashboard.py

If the raw tables have not been generated yet, the app generates them in-memory
from ``config.yaml`` on first load (nothing to set up).
"""
from __future__ import annotations

import sys
from pathlib import Path

# `streamlit run src/dashboard.py` executes this file as a top-level script, so
# the package context is lost and relative imports fail. Put the repo root on
# the path and use absolute imports, which also work when imported in tests.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd  # noqa: E402
import streamlit as st  # noqa: E402

from src import config as C  # noqa: E402
from src import data_io  # noqa: E402
from src import generate_data as G  # noqa: E402
from src import metrics as M  # noqa: E402
from src import report as R  # noqa: E402
from src import validate as V  # noqa: E402
from src import visualize as Vz  # noqa: E402


# ---------------------------------------------------------------------------
# Cached data / computation layer (computed once per session)
# ---------------------------------------------------------------------------
@st.cache_data(show_spinner="Loading synthetic data...")
def load_data() -> dict[str, pd.DataFrame]:
    """Load raw tables from disk, or generate them in-memory if absent."""
    try:
        return data_io.load_raw()
    except FileNotFoundError:
        return G.build_all(C.load_config())


@st.cache_data(show_spinner="Computing metrics...")
def metric_tables(exclude_incomplete: bool) -> dict[str, pd.DataFrame]:
    return M.build_metric_tables(load_data(), C.load_config(),
                                 exclude_incomplete=exclude_incomplete)


@st.cache_data(show_spinner="Running validation...")
def validation_summary() -> pd.DataFrame:
    return V.build_summary(V.run_validation(load_data(), C.load_config()))


@st.cache_data(show_spinner="Writing report...")
def report_markdown() -> str:
    return R.build_report(load_data(), C.load_config())


def _pct(x: float) -> str:
    return "n/a" if pd.isna(x) else f"{x * 100:.1f}%"


# ---------------------------------------------------------------------------
# Page
# ---------------------------------------------------------------------------
def main() -> None:
    cfg = C.load_config()
    target = float(cfg["target_growth_rate"])
    Vz._style()  # consistent chart styling

    st.set_page_config(
        page_title="Health Cost Growth Target Analytics",
        page_icon="📊", layout="wide",
    )

    # ---- sidebar ----
    st.sidebar.title("Cost Growth Target Program")
    st.sidebar.caption(
        f"Measurement years {cfg['years'][0]}–{cfg['years'][-1]} · "
        f"target {target:.1%}"
    )
    st.sidebar.warning(
        "**Synthetic data.** No real members, claims, providers, or payers. "
        "For skill demonstration only."
    )
    exclude_incomplete = st.sidebar.toggle(
        "Completeness-adjusted view", value=True,
        help="Exclude validation-flagged incomplete submissions from cost-"
             "growth trends (recommended).",
    )

    data = load_data()
    tables = metric_tables(exclude_incomplete)
    summary = validation_summary()
    exe = tables["executive_summary_metrics"]

    st.title("Health Cost Growth Target Analytics")

    # ---- headline KPIs ----
    latest = exe.dropna(subset=["yoy_growth"]).iloc[-1]
    n_fail = int((summary["status"] == "FAIL").sum())
    k1, k2, k3, k4, k5 = st.columns(5)
    k1.metric("Members", f"{data['enrollment']['member_id'].nunique():,}")
    k2.metric("Claim lines", f"{len(data['claims']):,}")
    k3.metric(f"PMPM ({cfg['years'][-1]})",
              f"${exe.iloc[-1]['truncated_pmpm']:,.0f}")
    k4.metric("Latest YoY growth", _pct(latest["yoy_growth"]),
              delta=f"{(latest['yoy_growth'] - target) * 100:+.1f} pts vs target",
              delta_color="inverse")
    k5.metric("Validation checks failed", f"{n_fail} / {len(summary)}")

    tabs = st.tabs([
        "Overview", "Payers", "Provider Orgs", "Service Categories",
        "High-Cost Members", "Data Quality", "Executive Summary",
    ])

    # ---- Overview ----
    with tabs[0]:
        c1, c2 = st.columns(2)
        c1.pyplot(Vz.fig_pmpm_by_year(exe))
        c2.pyplot(Vz.fig_growth_vs_target(exe, target))
        st.caption("Trend measured on high-cost-truncated allowed dollars "
                   f"(truncation ${cfg['truncation_threshold']:,.0f}).")

    # ---- Payers ----
    with tabs[1]:
        payer = tables["payer_cost_growth_summary"]
        st.pyplot(Vz.fig_payer_cost_growth(payer, target))
        view = payer.drop_duplicates("payer_id")[
            ["payer_id", "payer_name", "payer_type", "cagr_truncated",
             "cagr_exceeds_target"]
            + (["complete_submission"] if "complete_submission" in payer else [])
        ].rename(columns={"cagr_truncated": "cost_growth_cagr"})
        st.dataframe(view, use_container_width=True, hide_index=True)

    # ---- Providers ----
    with tabs[2]:
        provider = tables["provider_cost_growth_summary"]
        pooled = M.drop_payers(
            data, {p for p, _ in M.incomplete_payer_years(data, cfg)}
        ) if exclude_incomplete else data
        c1, c2 = st.columns([1, 1])
        c1.pyplot(Vz.fig_provider_cost_growth(provider, target))
        c2.pyplot(Vz.fig_payer_provider_heatmap(pooled, cfg))

        st.subheader("What's driving the highest-growth providers?")
        ctrl1, ctrl2 = st.columns([2, 3])
        top_n = ctrl1.slider("Providers to show (by overall cost growth)",
                             min_value=2, max_value=6, value=4)
        measure = ctrl2.radio(
            "Measure", ["Growth (%)", "PMPM $ change"], horizontal=True,
            help="% shows the rate; $ weights each category by size — a big "
                 "category like Inpatient dominates the dollars even at a "
                 "modest rate.",
        )
        value = "dollars" if measure == "PMPM $ change" else "growth"
        st.pyplot(Vz.fig_provider_category_growth(pooled, cfg, top_n=top_n,
                                                  value=value))
        st.caption(
            "Within each provider, by service category. **Growth (%)** is the "
            "rate; **PMPM $ change** is the actual dollar contribution, so you "
            "can see whether a provider's growth is broad-based or concentrated "
            "in a few big categories. Provider × category figures are volatile "
            "at this granularity — read as directional."
        )

    # ---- Service categories ----
    with tabs[3]:
        trends = tables["service_category_trends"]
        c1, c2 = st.columns(2)
        c1.pyplot(Vz.fig_service_category_growth(trends, cfg))
        c2.pyplot(Vz.fig_utilization_per_1000(trends, cfg))
        st.caption(
            "Claims with a missing service category are bucketed as "
            "*Unclassified* and excluded from these clinical-category charts; "
            "they are counted as a data-quality issue (DQ006) in the Data "
            "Quality tab."
        )

        st.subheader("Is it price or utilization?")
        pooled_sc = M.drop_payers(
            data, {p for p, _ in M.incomplete_payer_years(data, cfg)}
        ) if exclude_incomplete else data
        st.pyplot(Vz.fig_price_utilization(pooled_sc, cfg))
        st.caption(
            "PMPM = utilization × unit price, so growth splits into **more "
            "services** (utilization) vs **higher cost per service** (unit "
            "price). This distinguishes volume-driven growth from price-driven "
            "growth, which call for different interventions."
        )

    # ---- High-cost members ----
    with tabs[4]:
        conc = tables["high_cost_concentration"]
        pooled_hc = M.drop_payers(
            data, {p for p, _ in M.incomplete_payer_years(data, cfg)}
        ) if exclude_incomplete else data
        pooled_row = conc[conc["period"] == "All"].iloc[0]
        c1, c2, c3 = st.columns(3)
        c1.metric("Top 1% of members → spend share",
                  _pct(pooled_row["top_1pct_spend_share"]))
        c2.metric("Top 5% of members → spend share",
                  _pct(pooled_row["top_5pct_spend_share"]))
        c3.metric(f"Claims above ${cfg['truncation_threshold']:,.0f} → spend share",
                  _pct(pooled_row["high_cost_claim_spend_share"]))
        st.pyplot(Vz.fig_high_cost_concentration(pooled_hc, cfg))
        st.caption(
            "A small share of members drives a disproportionate share of spend "
            "— the rationale for high-cost truncation when measuring trend. "
            "Shares by year:"
        )
        st.dataframe(
            conc.rename(columns={
                "top_1pct_spend_share": "top_1%",
                "top_5pct_spend_share": "top_5%",
                "top_10pct_spend_share": "top_10%",
                "high_cost_claim_spend_share": "high_cost_claims",
            }),
            use_container_width=True, hide_index=True,
        )

        st.subheader("Where are the high-cost members?")
        hcf = M.high_cost_by_facility(
            M.prepare_analytic_claims(data, cfg), data["enrollment"], cfg
        ).merge(data["provider_org"][["provider_org_id", "provider_org_name"]],
                on="provider_org_id", how="left")
        lead = hcf.sort_values("high_cost_share", ascending=False).iloc[0]
        st.pyplot(Vz.fig_high_cost_by_facility(data, cfg))
        st.caption(
            f"**{lead['provider_org_name']}** holds "
            f"**{lead['high_cost_share']:.0%}** of high-cost members "
            f"({lead['concentration_index']:.1f}× its panel share) — a "
            f"tertiary/referral case-mix pattern. Its raw PMPM "
            f"(${lead['raw_pmpm']:,.0f}) runs well above its truncated PMPM "
            f"(${lead['truncated_pmpm']:,.0f}); the gap is catastrophic "
            f"case-mix, not inefficiency — which is exactly why fair provider "
            f"comparisons need **risk adjustment**."
        )

    # ---- Data quality ----
    with tabs[5]:
        n_claims = len(data["claims"])
        critical_records = int(
            summary.loc[summary["severity"] == "CRITICAL", "issue_count"].sum()
        )
        total_flagged = int(summary["issue_count"].sum())
        clean_critical = 1 - critical_records / n_claims
        claim_lvl = summary[summary["dimension"] == "claims"]
        found_cl = int(claim_lvl["issue_count"].sum())
        resolved_cl = int(claim_lvl["resolved_count"].sum())

        st.markdown(
            "Every submission is screened by a documented rule set. Issues are "
            "**detected, quantified, and excluded (or de-duplicated) before the "
            "cost analysis** — so the figures below are follow-up items for "
            "submitters, not blockers. Even the most severe checks affect well "
            "under 1% of claims."
        )
        m1, m2, m3 = st.columns(3)
        m1.metric("Claim lines clean of critical issues", f"{clean_critical:.1%}")
        m2.metric("Claim records resolved before analysis",
                  f"{resolved_cl:,} of {found_cl:,}")
        m3.metric("Checks failed", f"{n_fail} of {len(summary)}")

        c1, c2 = st.columns(2)
        c1.pyplot(Vz.fig_validation_by_severity(summary))
        c2.pyplot(Vz.fig_validation_resolution(summary))
        st.caption(
            "Duplicate claim IDs are removed (keep-first); negative / "
            "paid-over-allowed / orphan / invalid-date records are excluded from "
            "all cost metrics; high-cost claims are truncated; the incomplete "
            "payer-year is held out of trend results. The 'handling' column "
            "below shows each check's disposition."
        )
        st.subheader("Validation detail")
        st.dataframe(
            summary[["check_id", "check_name", "severity", "status",
                     "issue_count", "affected_pct", "handling",
                     "recommended_follow_up"]],
            use_container_width=True, hide_index=True,
        )

    # ---- Executive summary ----
    with tabs[6]:
        st.markdown(report_markdown())


if __name__ == "__main__":
    main()
