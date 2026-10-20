"""Render a one-page summary dashboard of the study from pipeline outputs.

Reads the committed outputs (data/industry_panel.csv, outputs/scores.csv,
outputs/forecasts.csv) - i.e. the exact published numbers - and composes a
single shareable figure:

    +--------------------------------------------------+
    | title / subtitle                                  |
    | A. composite ranking   | B. market trajectories  |
    | C. growth-scale map     | D. next top 5 + verdict|
    | footer: method + links                            |
    +--------------------------------------------------+

Usage:  python make_summary.py
Output: outputs/charts/00_summary_dashboard.png
        (+ a copy in the user's Downloads folder)
"""
from __future__ import annotations

import pathlib
import shutil

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import FancyBboxPatch

from src.visuals import INDUSTRY_COLORS, SHORT_NAMES

ROOT = pathlib.Path(__file__).resolve().parent
OUT_PNG = ROOT / "outputs" / "charts" / "00_summary_dashboard.png"
DOWNLOADS = pathlib.Path.home() / "Downloads"
DOWNLOAD_NAME = "Industry-ML-Prospects-Visualization.png"

TEAL, BLUE, GREEN, RED = "#00A499", "#1F6FB2", "#1E8449", "#C0392B"


def _color(name: str) -> str:
    return INDUSTRY_COLORS.get(name, "#7F8C8D")


def _despine(ax) -> None:
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)


def render() -> None:
    # ------------------------------------------------------------------ data
    panel = pd.read_csv(ROOT / "data" / "industry_panel.csv")
    scores = pd.read_csv(ROOT / "outputs" / "scores.csv", index_col=0)
    fc = pd.read_csv(ROOT / "outputs" / "forecasts.csv")
    mkt_fc = fc[fc["metric"] == "market_size_usd_b"].sort_values(["industry", "year"])

    industries = list(scores.index)  # already rank-ordered
    tiers = scores["tier"].to_dict()
    top5 = industries[:5]
    top5_set = set(top5)
    last_year = int(panel["year"].max())
    final_year = int(mkt_fc["year"].max())
    m2025 = panel[panel["year"] == last_year].set_index("industry")["market_size_usd_b"]

    # ----------------------------------------------------------------- style
    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "axes.edgecolor": "#D0D0D0",
            "axes.grid": True,
            "grid.color": "#EBEBEB",
            "grid.linewidth": 0.7,
            "axes.axisbelow": True,
            "font.size": 9,
            "xtick.color": "#666",
            "ytick.color": "#666",
            "axes.labelcolor": "#444",
            "text.color": "#222",
        }
    )

    fig = plt.figure(figsize=(16, 10), dpi=200)
    gs = fig.add_gridspec(
        nrows=2, ncols=2, left=0.055, right=0.975, top=0.865,
        bottom=0.105, hspace=0.30, wspace=0.20,
    )
    ax_rank = fig.add_subplot(gs[0, 0])
    ax_mkt = fig.add_subplot(gs[0, 1])
    ax_map = fig.add_subplot(gs[1, 0])
    ax_verd = fig.add_subplot(gs[1, 1])

    # ------------------------------------------------------------- header
    fig.suptitle(
        "The NEXT Top Industries for Data Science & Machine Learning",
        x=0.055, y=0.965, ha="left", fontsize=21, fontweight="bold",
    )
    fig.text(
        0.055, 0.918,
        f"A {final_year - 4}\u2013{final_year} forecast & composite ranking of 13 candidate industries "
        f"\u2014 5 incumbents vs 8 rising challengers",
        fontsize=11.5, color="#555",
    )

    # ------------------------------------------------- A. composite ranking
    df = scores.sort_values("score")
    n = len(df)
    labels = [
        "\u25B2 " + ind if tiers.get(ind, "") == "Challenger" else ind
        for ind in df.index
    ]
    pos = np.arange(n)
    ax_rank.axhspan(n - 5 - 0.5, n - 0.5, color=GREEN, alpha=0.08)
    bars = ax_rank.barh(
        pos, df["score"], color=[_color(i) for i in df.index], height=0.62,
    )
    for bar, val, ind in zip(bars, df["score"], df.index):
        bar.set_alpha(1.0 if ind in top5_set else 0.55)
        ax_rank.text(
            val + 1.4, bar.get_y() + bar.get_height() / 2, f"{val:.1f}",
            va="center", fontsize=8.2, fontweight="bold", color="#333",
        )
    ax_rank.set_yticks(pos)
    ax_rank.set_yticklabels(labels, fontsize=8.4)
    ax_rank.set_xlim(0, df["score"].max() * 1.15)
    ax_rank.set_ylim(-0.7, n - 0.3)
    ax_rank.set_xlabel("Composite prospect score (0\u2013100)", fontsize=9)
    ax_rank.grid(axis="y", visible=False)
    ax_rank.set_title(
        "A \u00b7 Composite prospect ranking \u2014 all 13 candidates",
        loc="left", fontsize=12.5, fontweight="bold",
    )
    ax_rank.text(
        df["score"].max() * 0.02, n - 3, "predicted next top 5",
        fontsize=8.5, fontweight="bold", color=GREEN, style="italic",
        va="center",
    )
    _despine(ax_rank)

    # --------------------------------------------- B. market trajectories
    for ind in industries:
        if ind in top5_set:
            continue
        sub = panel[panel["industry"] == ind].sort_values("year")
        fcm = mkt_fc[mkt_fc["industry"] == ind]
        ax_mkt.plot(
            sub["year"], sub["market_size_usd_b"],
            color=_color(ind), lw=1.0, alpha=0.40, zorder=2,
        )
        ax_mkt.plot(
            fcm["year"], fcm["point"],
            color=_color(ind), lw=1.0, ls="--", alpha=0.40, zorder=2,
        )
    for ind in top5:
        sub = panel[panel["industry"] == ind].sort_values("year")
        fcm = mkt_fc[mkt_fc["industry"] == ind]
        c = _color(ind)
        ax_mkt.plot(
            sub["year"], sub["market_size_usd_b"], color=c, lw=2.2, zorder=4,
            label=ind,
        )
        ax_mkt.plot(
            fcm["year"], fcm["point"], color=c, lw=2.0, ls="--", zorder=4,
        )
        ax_mkt.fill_between(
            fcm["year"], fcm["ci90_lo"], fcm["ci90_hi"],
            color=c, alpha=0.12, lw=0, zorder=1.5,
        )
        ax_mkt.annotate(
            f"${fcm['point'].iloc[-1]:,.0f}B",
            xy=(final_year, fcm["point"].iloc[-1]),
            xytext=(5, 0), textcoords="offset points",
            fontsize=8, fontweight="bold", color=c, va="center",
        )
    ax_mkt.axvline(last_year + 0.5, color="#BBB", ls=":", lw=1)
    ax_mkt.text(
        last_year + 0.6, ax_mkt.get_ylim()[1] * 0.97, "forecast",
        fontsize=7.5, color="#999", va="top",
    )
    ax_mkt.set_xlim(2014.6, final_year + 1.8)
    ax_mkt.set_xlabel("Year", fontsize=9)
    ax_mkt.set_ylabel("DS/ML market size ($B)", fontsize=9)
    ax_mkt.legend(loc="upper left", fontsize=7.6, frameon=False)
    ax_mkt.set_title(
        "B \u00b7 Market size \u2014 history & ensemble forecast (90% CI)",
        loc="left", fontsize=12.5, fontweight="bold",
    )
    _despine(ax_mkt)

    # ----------------------------------------------- C. growth-scale map
    x = scores["growth_momentum"].to_numpy() * 100
    y = scores["market_scale"].to_numpy()
    med_x, med_y = float(np.median(x)), float(np.median(y))
    ax_map.axvline(med_x, color="#BBB", ls="--", lw=1)
    ax_map.axhline(med_y, color="#BBB", ls="--", lw=1)
    for ind, row in scores.iterrows():
        xi = row["growth_momentum"] * 100
        yi = row["market_scale"]
        challenger = str(row["tier"]) == "Challenger"
        ax_map.scatter(
            xi, yi, s=70 + (row["score"] / 100.0) * 380,
            color=_color(ind), marker="^" if challenger else "o",
            alpha=0.85, edgecolor="white", linewidth=1.2, zorder=3,
        )
        ax_map.annotate(
            SHORT_NAMES.get(ind, ind), xy=(xi, yi),
            xytext=(6, 5), textcoords="offset points",
            fontsize=7.5, color="#333",
        )
    ax_map.set_yscale("log")
    ticks = [5, 10, 25, 50, 100, 250]
    ax_map.set_yticks(ticks)
    ax_map.set_yticklabels([f"${t}B" for t in ticks], fontsize=8)
    pad = (x.max() - x.min()) * 0.06
    ax_map.set_xlim(x.min() - pad, x.max() + pad * 3)
    ax_map.text(0.985, 0.97, "RISING STARS", transform=ax_map.transAxes,
                ha="right", va="top", fontsize=8, color="#B5B5B5", fontweight="bold")
    ax_map.text(0.015, 0.97, "HEAVYWEIGHTS", transform=ax_map.transAxes,
                ha="left", va="top", fontsize=8, color="#B5B5B5", fontweight="bold")
    ax_map.text(0.985, 0.05, "NICHE ROCKETS", transform=ax_map.transAxes,
                ha="right", va="bottom", fontsize=8, color="#B5B5B5", fontweight="bold")
    ax_map.text(0.015, 0.05, "SLOW LANE", transform=ax_map.transAxes,
                ha="left", va="bottom", fontsize=8, color="#B5B5B5", fontweight="bold")
    ax_map.set_xlabel("Forecast market CAGR, 2025 \u2192 2030 (%)", fontsize=9)
    ax_map.set_ylabel(f"Forecast {final_year} market size (log)", fontsize=9)
    handles = [
        Line2D([], [], color="#888", marker="o", ls="", ms=8,
               label="Incumbent"),
        Line2D([], [], color="#888", marker="^", ls="", ms=8,
               label="Challenger (\u25B2)"),
    ]
    ax_map.legend(handles=handles, loc="upper left", fontsize=8, frameon=False)
    ax_map.set_title(
        "C \u00b7 Growth\u2013scale map \u2014 momentum vs 2030 market",
        loc="left", fontsize=12.5, fontweight="bold",
    )
    _despine(ax_map)

    # ------------------------------------------------ D. verdicts panel
    ax_verd.set_xlim(0, 1)
    ax_verd.set_ylim(0, 1)
    ax_verd.axis("off")
    ax_verd.set_title(
        "D \u00b7 The predicted next top 5 \u2014 and the verdicts",
        loc="left", fontsize=12.5, fontweight="bold",
    )

    header = (
        f"{'INDUSTRY':<30}{'SCORE':>6}   {'2025':>7} {'2030':>8}   CAGR"
    )
    ax_verd.text(
        0.02, 0.955, header, family="monospace",
        fontsize=8.2, color="#888", fontweight="bold",
    )
    for i, ind in enumerate(top5):
        y = 0.885 - i * 0.075
        ax_verd.text(
            0.02, y, f"#{i + 1}", fontsize=9.5, fontweight="bold",
            color=_color(ind),
        )
        line = (
            f"{ind[:28]:<30}{scores.loc[ind, 'score']:>6.1f}   "
            f"${m2025[ind]:>5.1f}B ${scores.loc[ind, 'market_scale']:>6.1f}B   "
            f"+{scores.loc[ind, 'growth_momentum'] * 100:.1f}%"
        )
        ax_verd.text(
            0.075, y, line, family="monospace", fontsize=9.2, color="#222",
        )

    box_specs = [
        (0.42, "#E8F0FA", BLUE, "HOLDING TOP-5",
         "AI Infrastructure \u00b7 Healthcare"),
        (0.22, "#E4F6EC", GREEN, "BREAKING IN",
         "Climate Tech \u00b7 Legal Tech \u00b7 EdTech"),
        (0.02, "#FBEAE8", RED, "DROPPING OUT",
         "Defense/Cyber (#8) \u00b7 Finance (#12) \u00b7 Energy/Ind. (#13)"),
    ]
    for y0, face, edge, label, content in box_specs:
        ax_verd.add_patch(
            FancyBboxPatch(
                (0.02, y0), 0.96, 0.155, boxstyle="round,pad=0.012",
                facecolor=face, edgecolor=edge, lw=1.1,
                transform=ax_verd.transAxes, zorder=2, mutation_aspect=1.2,
            )
        )
        ax_verd.text(
            0.045, y0 + 0.078, label, fontsize=8.6, fontweight="bold",
            color=edge, va="center",
        )
        ax_verd.text(
            0.045, y0 + 0.036, content, fontsize=9.2, color="#222", va="center",
        )

    # ------------------------------------------------------------- footer
    fig.text(
        0.055, 0.048,
        "Method: backtest-weighted ensemble of log-linear OLS growth regression + damped Holt "
        "exponential smoothing (ETS(A,Ad,N)-style) \u00b7 90% residual-bootstrap CIs \u00b7 "
        "logistic S-curve adoption \u00b7 min-max composite index "
        "(growth 30 \u00b7 scale 25 \u00b7 investment 20 \u00b7 jobs 15 \u00b7 adoption 10)",
        fontsize=8.6, color="#777",
    )
    fig.text(
        0.055, 0.022,
        "github.com/danielosinor123-gif/ds-ml-industry-prospects \u00b7 "
        "danielaxv3-ds-ml-industry-prospects.static.hf.space \u00b7 "
        "benchmark-calibrated data (GVR, Statista/Precedence, McKinsey State of AI, US BLS)",
        fontsize=8.2, color="#999",
    )

    OUT_PNG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_PNG)
    plt.close(fig)

    if DOWNLOADS.exists():
        shutil.copy2(OUT_PNG, DOWNLOADS / DOWNLOAD_NAME)
        print(f"saved: outputs/charts/00_summary_dashboard.png ({OUT_PNG.stat().st_size/1e6:.2f} MB)")
        print(f"saved: Downloads/{DOWNLOAD_NAME}")
    else:
        print(f"saved: outputs/charts/00_summary_dashboard.png ({OUT_PNG.stat().st_size/1e6:.2f} MB)")


if __name__ == "__main__":
    render()
