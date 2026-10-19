"""Chart generation for the 13-industry candidate pool.

Visual strategy for a busy 13-series chart:
- the *predicted top-5* industries are emphasised (thick lines, CI bands,
  end-value annotations);
- every other candidate is drawn muted (thin, semi-transparent) so the
  "who's rising" story stays readable;
- the ranking chart marks challengers with a triangle tag and shades the
  predicted next-top-5 zone;
- a growth-scale quadrant map plots every candidate by momentum vs 2030
  market size (bubble = composite score, triangle = challenger).
"""
from __future__ import annotations

from pathlib import Path
import textwrap

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.ticker import NullFormatter

from .forecast import logistic_value
from .scoring import COMPONENT_LABELS, COMPONENT_WEIGHTS

INDUSTRY_COLORS: dict[str, str] = {
    # incumbents
    "Healthcare": "#00A499",
    "Finance": "#1F6FB2",
    "Defense & Cybersecurity": "#C0392B",
    "Energy & Industrials": "#E67E22",
    "AI Infrastructure": "#6C3483",
    # challengers
    "Retail & E-commerce": "#F1C40F",
    "Transportation & Logistics": "#2C7873",
    "Agriculture & Food Tech": "#8CBF26",
    "Telecom & Media": "#D81B60",
    "Climate & Sustainability Tech": "#27AE60",
    "Insurance": "#5D6D7E",
    "Education & EdTech": "#B58AD1",
    "Legal & Compliance Tech": "#D35400",
}

SHORT_NAMES: dict[str, str] = {
    "Healthcare": "Healthcare",
    "Finance": "Finance",
    "Defense & Cybersecurity": "Defense/Cyber",
    "Energy & Industrials": "Energy/Ind.",
    "AI Infrastructure": "AI Infra",
    "Retail & E-commerce": "Retail/E-comm",
    "Transportation & Logistics": "Transport/Log.",
    "Agriculture & Food Tech": "Agriculture",
    "Telecom & Media": "Telecom/Media",
    "Climate & Sustainability Tech": "Climate Tech",
    "Insurance": "Insurance",
    "Education & EdTech": "Education",
    "Legal & Compliance Tech": "Legal Tech",
}

METRIC_CHARTS = {
    "market_size_usd_b": {
        "file": "01_market_size_forecast.png",
        "title": "Data Science & ML Market Size by Candidate Industry",
        "ylabel": "DS/ML market size ($B)",
        "fmt": "${v:,.0f}B",
        "footer": (
            "Benchmark-calibrated estimates · ensemble forecast "
            "(exponential growth + damped Holt) · CI band + annotations shown for "
            "predicted top-5 only"
        ),
    },
    "investment_usd_b": {
        "file": "02_investment_forecast.png",
        "title": "DS/ML Investment & Funding by Candidate Industry",
        "ylabel": "DS/ML investment ($B)",
        "fmt": "${v:,.0f}B",
        "footer": (
            "Benchmark-calibrated estimates · ensemble forecast "
            "(exponential growth + damped Holt) · CI band + annotations shown for "
            "predicted top-5 only"
        ),
    },
    "job_postings_k": {
        "file": "03_job_postings_forecast.png",
        "title": "DS/ML Job Postings by Candidate Industry",
        "ylabel": "DS/ML job postings (thousands/yr)",
        "fmt": "{v:,.0f}k",
        "footer": (
            "Benchmark-calibrated estimates · ensemble forecast "
            "(exponential growth + damped Holt) · CI band + annotations shown for "
            "predicted top-5 only"
        ),
    },
}


def _style() -> None:
    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "axes.edgecolor": "#D0D0D0",
            "axes.grid": True,
            "grid.color": "#E8E8E8",
            "grid.linewidth": 0.8,
            "axes.axisbelow": True,
            "font.size": 10,
            "axes.titlesize": 13,
            "axes.titleweight": "bold",
            "axes.titlelocation": "left",
            "axes.labelsize": 10.5,
            "legend.frameon": False,
            "xtick.color": "#666666",
            "ytick.color": "#666666",
            "axes.labelcolor": "#333333",
            "text.color": "#222222",
        }
    )


def _despine(ax) -> None:
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)


def _footer(fig, text: str) -> None:
    fig.text(0.012, 0.008, text, fontsize=7.2, color="#999999", ha="left")


def _color(industry: str) -> str:
    return INDUSTRY_COLORS.get(industry, "#7F8C8D")


# --------------------------------------------------------------------------
# Forecast trajectory charts (13 candidates, top-5 emphasised)
# --------------------------------------------------------------------------
def plot_metric_forecast(
    results: dict, metric: str, outpath: Path, emphasized: set | None = None
) -> None:
    cfg = METRIC_CHARTS[metric]
    matching = [(ind, res) for (ind, m), res in results.items() if m == metric]
    if not matching:
        return

    _style()
    fig, ax = plt.subplots(figsize=(11.8, 6.8), dpi=170)

    first_year = matching[0][1].hist_years[0]
    last_hist_year = matching[0][1].hist_years[-1]
    boundary = last_hist_year + 0.5
    last_forecast_year = max(res.years[-1] for _, res in matching)

    # draw muted lines first, emphasised lines on top
    ordered = sorted(
        matching,
        key=lambda item: (item[0] in (emphasized or set()),),
    )

    for industry, res in ordered:
        color = _color(industry)
        emph = industry in emphasized if emphasized else True
        lw_h, alpha = (2.3, 1.0) if emph else (1.1, 0.45)
        ax.plot(
            res.hist_years, res.hist_values,
            color=color, lw=lw_h, alpha=alpha, label=industry,
            zorder=4 if emph else 2.5,
        )
        ax.plot(
            res.years, res.point,
            color=color, lw=(2.0 if emph else 1.0), ls="--",
            marker="o" if emph else None, ms=4.5,
            alpha=alpha, zorder=4 if emph else 2.5,
        )
        if emph:
            ax.fill_between(
                res.years, res.lo, res.hi,
                color=color, alpha=0.11, lw=0, zorder=1.5,
            )
            ax.annotate(
                cfg["fmt"].format(v=res.point[-1]),
                xy=(res.years[-1], res.point[-1]),
                xytext=(6, 0), textcoords="offset points",
                fontsize=8.5, fontweight="bold", color=color, va="center",
            )

    ax.axvline(boundary, color="#B5B5B5", ls=":", lw=1.1)
    ax.text(
        boundary + 0.08, ax.get_ylim()[1] * 0.985, "forecast \u2192",
        fontsize=8, color="#999999", va="top",
    )
    ax.set_xlim(first_year - 0.4, last_forecast_year + 1.8)
    ax.set_title(cfg["title"])
    ax.set_ylabel(cfg["ylabel"])
    ax.set_xlabel("Year")
    ax.legend(loc="upper left", fontsize=8, ncol=2, columnspacing=1.2)
    _despine(ax)

    fig.tight_layout(rect=(0, 0.035, 1, 1))
    _footer(fig, cfg["footer"])
    fig.savefig(outpath)
    plt.close(fig)


# --------------------------------------------------------------------------
# Adoption S-curves (13 candidates, top-5 emphasised)
# --------------------------------------------------------------------------
def plot_adoption(
    panel: pd.DataFrame,
    adoption_fits: dict,
    last_hist_year: int,
    final_year: int,
    outpath: Path,
    emphasized: set | None = None,
) -> None:
    _style()
    fig, ax = plt.subplots(figsize=(11.8, 6.8), dpi=170)

    first_year = int(panel["year"].min())
    grid = np.arange(first_year, final_year + 0.26, 0.25)
    boundary = last_hist_year + 0.5

    ordered = sorted(
        adoption_fits.items(),
        key=lambda item: item[0] in (emphasized or set()),
    )

    for industry, (ceiling, k, midpoint, _) in ordered:
        color = _color(industry)
        emph = industry in emphasized if emphasized else True
        sub = panel[panel["industry"] == industry].sort_values("year")
        ax.scatter(
            sub["year"], sub["adoption_rate_pct"],
            s=(26 if emph else 14), color=color,
            alpha=(1.0 if emph else 0.5), zorder=4 if emph else 2.5,
        )
        curve = np.array([logistic_value(ceiling, k, midpoint, g) for g in grid])
        hist_mask = grid <= last_hist_year
        ax.plot(
            grid[hist_mask], curve[hist_mask],
            color=color, lw=(1.9 if emph else 1.0),
            alpha=(1.0 if emph else 0.45), zorder=3 if emph else 2,
        )
        ax.plot(
            grid[~hist_mask], curve[~hist_mask],
            color=color, lw=(1.6 if emph else 0.9), ls="--",
            alpha=(1.0 if emph else 0.45), zorder=3 if emph else 2,
        )
        if emph:
            final_val = logistic_value(ceiling, k, midpoint, final_year)
            ax.annotate(
                f"{final_val:.0f}%",
                xy=(final_year, final_val),
                xytext=(6, 0), textcoords="offset points",
                fontsize=8.5, fontweight="bold", color=color, va="center",
            )

    ax.axvline(boundary, color="#B5B5B5", ls=":", lw=1.1)
    ax.text(boundary + 0.08, 97, "projection \u2192", fontsize=8,
            color="#999999", va="top")
    ax.set_xlim(first_year - 0.4, final_year + 1.8)
    ax.set_ylim(0, 100)
    ax.set_title("ML Adoption by Candidate Industry — Logistic S-Curves")
    ax.set_ylabel("Firms deploying ML in production (%)")
    ax.set_xlabel("Year")
    ax.legend(
        [Line2D([], [], color=_color(ind), lw=2) for ind in adoption_fits],
        list(adoption_fits.keys()),
        loc="lower right", fontsize=8, ncol=2, columnspacing=1.2,
    )
    _despine(ax)

    fig.tight_layout(rect=(0, 0.035, 1, 1))
    _footer(fig, "Benchmark-calibrated estimates · logistic (S-curve) fits · "
                 "top-5 candidates emphasised")
    fig.savefig(outpath)
    plt.close(fig)


# --------------------------------------------------------------------------
# Composite ranking (all candidates, tier-aware, top-5 zone)
# --------------------------------------------------------------------------
def plot_ranking(
    scores: pd.DataFrame,
    outpath: Path,
    final_year: int,
    horizon: int = 5,
    top_n: int = 5,
) -> None:
    _style()
    df = scores.sort_values("score")  # ascending → best ends up at the top
    n = len(df)
    fig, ax = plt.subplots(figsize=(11.0, 7.2), dpi=170)

    labels = []
    for ind, row in df.iterrows():
        tag = "\u25B2 " if str(row.get("tier", "")) == "Challenger" else ""
        labels.append(f"{tag}{ind}")

    positions = np.arange(n)
    emph_mask = np.zeros(n, dtype=bool)
    emph_mask[n - top_n:] = True  # top-N are the last bars (highest scores)

    # shaded "predicted next top-N" zone
    ax.axhspan(n - top_n - 0.5, n - 0.5, color="#27AE60", alpha=0.07, zorder=1)
    ax.text(
        df["score"].max() * 0.03, n - top_n / 2 - 0.5,
        "predicted next top 5",
        fontsize=9, fontweight="bold", color="#1E8449",
        va="center", style="italic",
    )

    bars = ax.barh(
        positions, df["score"],
        color=[_color(ind) for ind in df.index],
        height=0.62, zorder=3,
    )
    for bar, val, emph in zip(bars, df["score"], emph_mask):
        bar.set_alpha(1.0 if emph else 0.55)
        ax.text(
            val + 1.4, bar.get_y() + bar.get_height() / 2, f"{val:.1f}",
            va="center", fontsize=8.6, fontweight="bold", color="#333333",
        )

    ax.set_yticks(positions)
    ax.set_yticklabels(labels, fontsize=9.5)
    ax.set_xlim(0, df["score"].max() * 1.14)
    ax.set_ylim(-0.7, n - 0.3)
    ax.set_xlabel("Composite prospect score (0–100)")
    ax.set_title(
        f"Composite DS/ML Prospect Ranking — Next Top Industries, "
        f"{final_year - horizon + 1}–{final_year}"
    )
    ax.grid(axis="y", visible=False)
    _despine(ax)

    fig.tight_layout(rect=(0, 0.05, 1, 1))
    weight_txt = " · ".join(
        f"{COMPONENT_LABELS[c].split(' (')[0].lower()} {int(w * 100)}%"
        for c, w in COMPONENT_WEIGHTS.items()
    )
    _footer(
        fig,
        f"\u25B2 = challenger (outside today's top-5) · green zone = predicted "
        f"next top-5 · score = weighted min-max blend · {weight_txt}",
    )
    fig.savefig(outpath)
    plt.close(fig)


# --------------------------------------------------------------------------
# Industry profile radar (predicted top-5 only)
# --------------------------------------------------------------------------
def plot_radar(scores: pd.DataFrame, outpath: Path) -> None:
    _style()
    components = list(COMPONENT_WEIGHTS.keys())
    labels = [textwrap.fill(COMPONENT_LABELS[c], 16) for c in components]
    n = len(components)
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False).tolist()
    angles_closed = angles + [angles[0]]

    fig = plt.figure(figsize=(9.5, 8.0), dpi=170)
    ax = fig.add_subplot(111, polar=True)

    for industry, row in scores.iterrows():
        vals = [float(row[f"{c}_norm"]) for c in components]
        vals_closed = vals + [vals[0]]
        color = _color(industry)
        ax.plot(angles_closed, vals_closed, color=color, lw=2.1, label=industry)
        ax.fill(angles_closed, vals_closed, color=color, alpha=0.10)

    ax.set_xticks(angles)
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylim(0, 1)
    ax.set_yticks([0.25, 0.5, 0.75, 1.0])
    ax.set_yticklabels(["0.25", "0.50", "0.75", "1.00"], fontsize=7.5, color="#999999")
    ax.grid(color="#DDDDDD")
    ax.spines["polar"].set_color("#DDDDDD")
    ax.set_title(
        "Profiles of the Predicted Next Top 5 (normalised components, 0–1)",
        fontsize=13, fontweight="bold", pad=26,
    )
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.05), ncol=2, fontsize=9)

    fig.tight_layout(rect=(0, 0.06, 1, 1))
    _footer(fig, "Each axis is min-max normalised across all 13 candidates")
    fig.savefig(outpath)
    plt.close(fig)


# --------------------------------------------------------------------------
# Growth–scale quadrant map (all candidates)
# --------------------------------------------------------------------------
def plot_growth_scale_map(
    scores: pd.DataFrame, outpath: Path, final_year: int
) -> None:
    _style()
    fig, ax = plt.subplots(figsize=(11.8, 7.4), dpi=170)

    x = scores["growth_momentum"].to_numpy() * 100.0
    y = scores["market_scale"].to_numpy()
    med_x, med_y = float(np.median(x)), float(np.median(y))

    ax.axvline(med_x, color="#BBBBBB", ls="--", lw=1.0, zorder=1)
    ax.axhline(med_y, color="#BBBBBB", ls="--", lw=1.0, zorder=1)

    for industry, row in scores.iterrows():
        xi = row["growth_momentum"] * 100.0
        yi = row["market_scale"]
        challenger = str(row.get("tier", "")) == "Challenger"
        marker = "^" if challenger else "o"
        size = 90 + (row["score"] / 100.0) * 520
        ax.scatter(
            xi, yi, s=size, color=_color(industry), marker=marker,
            alpha=0.85, edgecolor="white", linewidth=1.4, zorder=3,
        )
        ax.annotate(
            SHORT_NAMES.get(industry, industry),
            xy=(xi, yi), xytext=(7, 6), textcoords="offset points",
            fontsize=8, color="#333333",
        )

    # quadrant labels
    ax.text(0.985, 0.97, "RISING STARS\nhigh growth × big scale",
            transform=ax.transAxes, ha="right", va="top",
            fontsize=8.5, color="#B5B5B5", fontweight="bold")
    ax.text(0.015, 0.97, "HEAVYWEIGHTS\nbig scale, slower growth",
            transform=ax.transAxes, ha="left", va="top",
            fontsize=8.5, color="#B5B5B5", fontweight="bold")
    ax.text(0.985, 0.05, "NICHE ROCKETS\nfast but small",
            transform=ax.transAxes, ha="right", va="bottom",
            fontsize=8.5, color="#B5B5B5", fontweight="bold")
    ax.text(0.015, 0.05, "SLOW LANE",
            transform=ax.transAxes, ha="left", va="bottom",
            fontsize=8.5, color="#B5B5B5", fontweight="bold")

    ax.set_yscale("log")
    ticks = [5, 10, 25, 50, 100, 250]
    ax.set_yticks(ticks)
    ax.set_yticklabels([f"${t}B" for t in ticks], fontsize=9)
    ax.yaxis.set_minor_formatter(NullFormatter())

    pad = (x.max() - x.min()) * 0.06
    ax.set_xlim(x.min() - pad, x.max() + pad * 3)
    ax.set_title(f"The Growth–Scale Map — {final_year} DS/ML Candidates")
    ax.set_xlabel("Forecast market CAGR, 2025 → 2030 (%)")
    ax.set_ylabel(f"Forecast {final_year} market size (log scale)")

    handles = [
        Line2D([], [], color="#888888", marker="o", ls="", ms=9,
               label="Incumbent (today's top-5)"),
        Line2D([], [], color="#888888", marker="^", ls="", ms=9,
               label="Challenger (rising)"),
    ]
    ax.legend(handles=handles, loc="upper left", fontsize=9)
    _despine(ax)

    fig.tight_layout(rect=(0, 0.04, 1, 1))
    _footer(fig, "Bubble size = composite prospect score · dashed lines = "
                 "median growth / median 2030 scale")
    fig.savefig(outpath)
    plt.close(fig)
