"""Pipeline entry point.

Runs the full study end-to-end:
    1. build (or load) the candidate-pool panel (incumbents + challengers)
    2. ensemble-forecast market size, investment and job postings
    3. fit logistic S-curves to ML adoption and project forward
    4. compute the composite prospect index across ALL candidates
    5. render all charts (predicted top-5 emphasised)
    6. write the markdown report (who holds / breaks in / drops out)

Usage:
    python main.py
    python main.py --from-csv data/industry_panel.csv
    python main.py --horizon 8 --seed 7
"""
from __future__ import annotations

import argparse
import pathlib
import sys

import pandas as pd

from src import visuals
from src.data import build_panel, load_panel_from_csv
from src.forecast import ensemble_forecast, fit_logistic, logistic_value
from src.report import generate_report
from src.scoring import build_score_table

FORECAST_METRICS = ["market_size_usd_b", "investment_usd_b", "job_postings_k"]
TOP_N = 5

ROOT = pathlib.Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
OUT_DIR = ROOT / "outputs"
CHART_DIR = OUT_DIR / "charts"
REPORT_DIR = OUT_DIR / "reports"


def _banner(title: str) -> None:
    print("\n" + "=" * 72)
    print(title)
    print("=" * 72)


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Rank the NEXT top industries for DS/ML (incumbents vs challengers)."
    )
    p.add_argument("--seed", type=int, default=42, help="RNG seed (default 42)")
    p.add_argument("--horizon", type=int, default=5, help="Forecast horizon in years")
    p.add_argument(
        "--from-csv", type=str, default=None,
        help="Optional path to a panel CSV (same schema as data/industry_panel.csv)",
    )
    return p.parse_args(argv)


def _fmt_b(v: float) -> str:
    return f"${v:,.1f}B"


def _enable_utf8_console() -> str:
    """Make the console UTF-8 friendly; return a safe challenger marker."""
    try:
        if sys.stdout.encoding and sys.stdout.encoding.lower() not in ("utf-8", "utf8"):
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
    enc = (getattr(sys.stdout, "encoding", "") or "").lower()
    return "\u25B2" if "utf" in enc else "*"


def main(argv=None) -> int:
    args = parse_args(argv)
    challenger_mark = _enable_utf8_console()
    for d in (DATA_DIR, CHART_DIR, REPORT_DIR):
        d.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------- 1. data
    _banner("[1/6] Building the candidate-pool panel dataset")
    if args.from_csv:
        panel = load_panel_from_csv(args.from_csv)
        print(f"  loaded external panel: {args.from_csv}")
    else:
        panel = build_panel(seed=args.seed)
    n_ind = panel["industry"].nunique()
    n_tiers = panel["tier"].value_counts().to_dict() if "tier" in panel else {}
    print(f"  {n_ind} candidate industries: {n_tiers}")
    panel.to_csv(DATA_DIR / "industry_panel.csv", index=False)
    print(f"  saved: data/industry_panel.csv  ({len(panel)} rows)")

    industries = list(panel["industry"].unique())
    tiers = {
        ind: str(panel.loc[panel["industry"] == ind, "tier"].iloc[0])
        for ind in industries
    }
    last_year = int(panel["year"].max())
    final_year = last_year + args.horizon

    # -------------------------------------------------------- 2. forecasting
    _banner(f"[2/6] Ensemble forecasting {last_year + 1}-{final_year} (all candidates)")
    results: dict = {}
    seed_counter = 0
    for industry in industries:
        sub = panel[panel["industry"] == industry].sort_values("year")
        for metric in FORECAST_METRICS:
            seed_counter += 1
            results[(industry, metric)] = ensemble_forecast(
                industry=industry,
                metric=metric,
                years=sub["year"].to_numpy(),
                values=sub[metric].to_numpy(),
                horizon=args.horizon,
                seed=args.seed * 1000 + seed_counter,
            )
        res = results[(industry, "market_size_usd_b")]
        print(
            f"  {industry:<32} market {_fmt_b(res.hist_values[-1])} -> "
            f"{_fmt_b(res.point[-1])} ({res.horizon_cagr:+.1%} CAGR)"
        )

    # ------------------------------------------------------ 3. adoption curves
    _banner("[3/6] Fitting ML adoption S-curves")
    adoption_fits: dict = {}
    adoption_proj: dict = {}
    adoption_2025: dict = {}
    for industry in industries:
        sub = panel[panel["industry"] == industry].sort_values("year")
        ceiling, k, midpoint, sse = fit_logistic(
            sub["year"].to_numpy(), sub["adoption_rate_pct"].to_numpy()
        )
        adoption_fits[industry] = (ceiling, k, midpoint, sse)
        adoption_proj[industry] = logistic_value(ceiling, k, midpoint, final_year)
        adoption_2025[industry] = float(sub.iloc[-1]["adoption_rate_pct"])
    print(f"  fitted logistic curves for {len(adoption_fits)} candidates")

    # ------------------------------------------------------------ 4. scoring
    _banner(f"[4/6] Composite prospect index across {len(industries)} candidates")
    raw = {}
    for industry in industries:
        raw[industry] = {
            "growth_momentum": results[(industry, "market_size_usd_b")].horizon_cagr,
            "market_scale": float(results[(industry, "market_size_usd_b")].point[-1]),
            "investment_momentum": results[(industry, "investment_usd_b")].horizon_cagr,
            "job_demand": results[(industry, "job_postings_k")].horizon_cagr,
            "ai_maturity": adoption_proj[industry],
        }
    scores = build_score_table(raw, tiers=tiers)
    scores.to_csv(OUT_DIR / "scores.csv")

    forecast_rows = []
    for (industry, metric), res in results.items():
        for i, year in enumerate(res.years):
            forecast_rows.append(
                {
                    "industry": industry, "metric": metric, "year": year,
                    "point": round(float(res.point[i]), 3),
                    "ci90_lo": round(float(res.lo[i]), 3),
                    "ci90_hi": round(float(res.hi[i]), 3),
                }
            )
    pd.DataFrame(forecast_rows).to_csv(OUT_DIR / "forecasts.csv", index=False)
    print("  saved: outputs/scores.csv, outputs/forecasts.csv")

    top5 = list(scores.index[:TOP_N])
    top5_set = set(top5)

    # ------------------------------------------------------------ 5. charts
    _banner("[5/6] Rendering charts")
    for metric in FORECAST_METRICS:
        path = CHART_DIR / visuals.METRIC_CHARTS[metric]["file"]
        visuals.plot_metric_forecast(results, metric, path, emphasized=top5_set)
        print(f"  saved: outputs/charts/{path.name}")
    visuals.plot_adoption(
        panel, adoption_fits, last_year, final_year,
        CHART_DIR / "04_adoption_curves.png", emphasized=top5_set,
    )
    print("  saved: outputs/charts/04_adoption_curves.png")
    visuals.plot_ranking(
        scores, CHART_DIR / "05_composite_ranking.png",
        final_year, horizon=args.horizon, top_n=TOP_N,
    )
    print("  saved: outputs/charts/05_composite_ranking.png")
    visuals.plot_radar(
        scores.head(TOP_N), CHART_DIR / "06_industry_profiles_radar.png"
    )
    print("  saved: outputs/charts/06_industry_profiles_radar.png")
    visuals.plot_growth_scale_map(
        scores, CHART_DIR / "07_growth_scale_map.png", final_year
    )
    print("  saved: outputs/charts/07_growth_scale_map.png")

    # ------------------------------------------------------------ 6. report
    _banner("[6/6] Writing the markdown report")
    report_path = REPORT_DIR / "summary_report.md"
    generate_report(
        panel=panel,
        results=results,
        scores=scores,
        adoption_fits=adoption_fits,
        adoption_proj=adoption_proj,
        adoption_2025=adoption_2025,
        outpath=report_path,
        horizon=args.horizon,
        last_year=last_year,
        final_year=final_year,
        top_n=TOP_N,
    )
    print("  saved: outputs/reports/summary_report.md")

    # ------------------------------------------------------------- summary
    _banner(
        f"RESULT — The predicted NEXT top {TOP_N} industries for DS/ML "
        f"({last_year + 1}-{final_year})"
    )
    for industry, row in scores.iterrows():
        marker = challenger_mark if row.get("tier", "") == "Challenger" else " "
        print(
            f"  {marker} #{int(row['rank']):<2} {industry:<33} "
            f"score {row['score']:5.1f}   "
            f"(CAGR {row['growth_momentum']:+.1%}, "
            f"{final_year} market {_fmt_b(row['market_scale'])})"
        )

    held = [i for i in top5 if tiers[i] != "Challenger"]
    broke_in = [i for i in top5 if tiers[i] == "Challenger"]
    dropped = [
        i for i in industries if tiers[i] != "Challenger" and i not in top5
    ]
    print()
    print(f"  Incumbents holding top-5:  {', '.join(held) or '—'}")
    print(f"  Challengers breaking in:   {', '.join(broke_in) or '—'}")
    print(f"  Incumbents dropping out:   {', '.join(dropped) or '—'}")
    print()
    print("  Outputs:")
    print("    data/industry_panel.csv          - input panel dataset")
    print("    outputs/forecasts.csv             - all point forecasts + 90% CI")
    print("    outputs/scores.csv                - composite scores & components")
    print("    outputs/charts/*.png              - 7 publication-ready charts")
    print("    outputs/reports/summary_report.md - full written report")
    return 0


if __name__ == "__main__":
    sys.exit(main())
