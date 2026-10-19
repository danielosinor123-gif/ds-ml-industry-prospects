"""Auto-generated markdown report: the NEXT top DS/ML industries, ranked.

Framing: the candidate pool contains today's top-5 industries (incumbents)
plus eight rising challengers. The pipeline forecasts every candidate's
2026-2030 trajectory, scores them on a composite index, and answers:

    1. Which industries form the predicted NEXT top-5 for DS/ML?
    2. Which incumbents hold their spot, which drop out?
    3. Which challengers break in?
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from .scoring import COMPONENT_LABELS, strongest_component


def _md_table(headers: list[str], rows: list[list[str]]) -> str:
    """Small dependency-free GitHub-flavoured markdown table writer."""
    widths = [
        max(len(str(h)), *(len(str(r[i])) for r in rows)) if rows else len(str(h))
        for i, h in enumerate(headers)
    ]

    def fmt_row(cells: list) -> str:
        return "| " + " | ".join(
            f"{str(c):<{widths[i]}}" for i, c in enumerate(cells)
        ) + " |"

    sep = "| " + " | ".join("-" * w for w in widths) + " |"
    lines = [fmt_row(headers), sep] + [fmt_row(r) for r in rows]
    return "\n".join(lines)


def _pct(x: float) -> str:
    return f"{x * 100:.1f}%"


def _money(v: float) -> str:
    return f"${v:,.1f}B"


def _fmt_component(comp: str, value: float) -> str:
    if comp in ("growth_momentum", "investment_momentum", "job_demand"):
        return _pct(value)
    if comp == "market_scale":
        return _money(value)
    return f"{value:.1f}%"


def generate_report(
    panel: pd.DataFrame,
    results: dict,
    scores: pd.DataFrame,
    adoption_fits: dict,
    adoption_proj: dict,
    adoption_2025: dict,
    outpath: Path,
    horizon: int,
    last_year: int,
    final_year: int,
    top_n: int = 5,
) -> None:
    industries = list(scores.index)
    tiers = {
        ind: str(row.get("tier", "")) for ind, row in scores.iterrows()
    }
    top5 = list(scores.index[:top_n])

    # ------------------------------------------------ hold / break-in / drop
    held = [i for i in top5 if tiers.get(i) != "Challenger"]
    broke_in = [i for i in top5 if tiers.get(i) == "Challenger"]
    incumbents = [i for i in industries if tiers.get(i) != "Challenger"]
    dropped = [i for i in incumbents if i not in top5]

    winner = top5[0]
    w_row = scores.loc[winner]
    mkt = results[(winner, "market_size_usd_b")]

    def fmt_list(names: list[str]) -> str:
        return ", ".join(f"**{n}**" for n in names) if names else "—"

    # ---------------------------------------------------------------- tables
    rank_rows = []
    for ind, row in scores.iterrows():
        tier_tag = str(row.get("tier", ""))
        if tier_tag == "Challenger":
            tier_tag = "\u25B2 challenger"
        rank_rows.append(
            [
                str(int(row["rank"])), f"{ind} ({tier_tag})", f"{row['score']:.1f}",
                _pct(row["growth_momentum"]), _money(row["market_scale"]),
                _pct(row["investment_momentum"]), _pct(row["job_demand"]),
                f"{row['ai_maturity']:.1f}%",
            ]
        )
    ranking_table = _md_table(
        ["#", "Industry (tier)", "Score", "Market CAGR", f"{final_year} market",
         "Invest. CAGR", "Jobs CAGR", f"{final_year} adoption"],
        rank_rows,
    )

    market_rows = []
    for ind in industries:
        res = results[(ind, "market_size_usd_b")]
        market_rows.append(
            [
                ind,
                _money(float(res.hist_values[-1])),
                _money(float(res.point[-1])),
                f"[{_money(float(res.lo[-1]))} – {_money(float(res.hi[-1]))}]",
                _pct(res.horizon_cagr),
            ]
        )
    market_table = _md_table(
        ["Industry", f"{last_year} actual", f"{final_year} forecast",
         "90% CI", "Implied CAGR"],
        market_rows,
    )

    invest_rows = []
    for ind in industries:
        res = results[(ind, "investment_usd_b")]
        invest_rows.append(
            [
                ind,
                _money(float(res.hist_values[-1])),
                _money(float(res.point[-1])),
                _pct(res.horizon_cagr),
            ]
        )
    invest_table = _md_table(
        ["Industry", f"{last_year} actual", f"{final_year} forecast", "Implied CAGR"],
        invest_rows,
    )

    jobs_rows = []
    for ind in industries:
        res = results[(ind, "job_postings_k")]
        jobs_rows.append(
            [
                ind,
                f"{float(res.hist_values[-1]):,.0f}k",
                f"{float(res.point[-1]):,.0f}k",
                _pct(res.horizon_cagr),
            ]
        )
    jobs_table = _md_table(
        ["Industry", f"{last_year} actual", f"{final_year} forecast", "Implied CAGR"],
        jobs_rows,
    )

    adopt_rows = []
    for ind in industries:
        ceiling, _, _, sse = adoption_fits[ind]
        adopt_rows.append(
            [
                ind,
                f"{adoption_2025[ind]:.1f}%",
                f"{adoption_proj[ind]:.1f}%",
                f"{ceiling:.1f}%",
                f"{sse:.1f}",
            ]
        )
    adopt_table = _md_table(
        ["Industry", f"{last_year} adoption", f"{final_year} projection",
         "Fitted ceiling", "Fit SSE"],
        adopt_rows,
    )

    pool_rows = []
    for ind in industries:
        res = results[(ind, "market_size_usd_b")]
        pool_rows.append(
            [
                ind,
                str(scores.loc[ind].get("tier", "")),
                _money(float(res.hist_values[-1])),
                f"{adoption_2025[ind]:.1f}%",
            ]
        )
    pool_table = _md_table(
        ["Industry", "Tier", f"{last_year} market", f"{last_year} adoption"],
        pool_rows,
    )

    # ---------------------------------------------------------------- snapshots
    snapshots = []
    for ind in top5:
        row = scores.loc[ind]
        comp = strongest_component(row)
        tier_note = (
            " *(challenger breaking in)*" if tiers.get(ind) == "Challenger" else ""
        )
        snapshots.append(
            f"- **{ind}** — rank **#{int(row['rank'])}**, score "
            f"**{row['score']:.1f}**{tier_note}. Strongest dimension: "
            f"{COMPONENT_LABELS[comp].lower()} "
            f"({_fmt_component(comp, row[comp])})."
        )

    # highest-ranked challenger context
    best_challenger = next(
        (i for i in top5 if tiers.get(i) == "Challenger"), None
    )
    if best_challenger:
        bc_res = results[(best_challenger, "market_size_usd_b")]
        bc_rank = int(scores.loc[best_challenger]["rank"])
        rising_para = (
            f"The highest-ranked challenger is **{best_challenger}** "
            f"(#{bc_rank} overall): its DS/ML market is forecast to grow from "
            f"{_money(float(bc_res.hist_values[-1]))} to "
            f"{_money(float(bc_res.point[-1]))} by {final_year} "
            f"({_pct(bc_res.horizon_cagr)} CAGR) — out-pacing several larger, "
            f"more mature incumbents on momentum."
        )
    else:
        rising_para = (
            "No challenger displaces an incumbent inside the predicted top-5 "
            "under the current weights — though several challengers post faster "
            "growth from smaller bases (see the growth–scale map)."
        )

    weight_lines = "\n".join(
        f"| {COMPONENT_LABELS[c]} | {int(w * 100)}% |"
        for c, w in
        [
            ("growth_momentum", 0.30), ("market_scale", 0.25),
            ("investment_momentum", 0.20), ("job_demand", 0.15),
            ("ai_maturity", 0.10),
        ]
    )

    n_cand = len(industries)
    n_chal = sum(1 for i in industries if tiers.get(i) == "Challenger")

    md = f"""# The NEXT Top Industries for Data Science & Machine Learning
### A {last_year + 1}–{final_year} Composite Ranking of {n_cand} Candidate Industries

*Generated automatically by the pipeline in `main.py` · panel covers
{int(panel['year'].min())}–{last_year} · forecast horizon: {horizon} years ·
{n_cand - n_chal} incumbents vs {n_chal} challengers*

---

## 1. Executive Summary

The predicted **next top-5 industries for DS/ML ({last_year + 1}–{final_year})**:

{chr(10).join(f"{i + 1}. **{ind}** — score {scores.loc[ind, 'score']:.1f}" for i, ind in enumerate(top5))}

**Incumbents holding their spot:** {fmt_list(held)}
**Challengers breaking in:** {fmt_list(broke_in)}
**Incumbents dropping out:** {fmt_list(dropped)}

**{winner}** leads the index with a score of **{w_row['score']:.1f}/100**: its
DS/ML market is projected to grow from {_money(float(mkt.hist_values[-1]))}
({last_year}) to **{_money(float(mkt.point[-1]))} by {final_year}** — a
**{_pct(mkt.horizon_cagr)} CAGR**.

{rising_para}

**The question this project answers:** *beyond today's recognised top-5
industries for DS/ML, which industries rise next — and do any of them
displace the incumbents?*

---

## 2. The Candidate Pool

The pool pairs today's recognised top-5 (incumbents, from the original
research list) with eight rising challengers chosen for credible DS/ML
momentum: retail personalisation, logistics optimisation, precision
agriculture, telecom AI, climate analytics, insurtech, edtech and legal AI.

{pool_table}

Adding a candidate is a single entry in `INDUSTRY_CALIBRATION`
(`src/data.py`) — the whole pipeline re-runs unchanged.

---

## 3. Methodology

**Data.** An industry × year panel ({n_cand} industries ×
{last_year - int(panel['year'].min()) + 1} years, {len(panel)} observations)
tracking market size, investment, job postings, ML adoption and patents.
Values are benchmark-calibrated to published estimates (see §12); the
pipeline accepts any real CSV with the same schema via `--from-csv`.

**Forecasting.** Each unbounded metric is forecast with a two-model ensemble:

1. **Exponential growth** — OLS on log(y) ~ t (compound-growth model);
2. **Damped Holt trend** — Holt's linear trend with damping to prevent
   runaway extrapolation.

Model weights are learned by **backtesting on a 3-year holdout** (weight ∝
1/MAPE). The 90% confidence band is a **mixture bootstrap**: 1,000 simulated
paths, each drawn from a component model with probability equal to its
weight. Adoption rates are bounded, so they are fit with a **logistic
S-curve** (grid-searched ceiling, closed-form logit regression).

**Composite scoring.** Components are min-max normalised **across all
{n_cand} candidates** and blended:

| Component | Weight |
|---|---|
{weight_lines}

---

## 4. Full Ranking ({last_year + 1}–{final_year})

{ranking_table}

![Composite ranking](../charts/05_composite_ranking.png)

---

## 5. Market Size Forecasts

{market_table}

![Market size forecast](../charts/01_market_size_forecast.png)

## 6. Investment & Funding Forecasts

{invest_table}

![Investment forecast](../charts/02_investment_forecast.png)

## 7. Job Demand Forecasts

{jobs_table}

![Job postings forecast](../charts/03_job_postings_forecast.png)

## 8. ML Adoption Projection

{adopt_table}

![Adoption curves](../charts/04_adoption_curves.png)

---

## 9. The Predicted Next Top 5 — Snapshots

{chr(10).join(snapshots)}

![Profiles of the next top 5](../charts/06_industry_profiles_radar.png)

## 10. Momentum vs Scale — Reading the Trade-off

This index is deliberately **momentum-weighted** (growth 30% + investment
20% + jobs 15% vs scale 25% + adoption 10%). That means smaller-but-faster
niches can outrank larger-but-slower incumbents on *prospect*, which is the
point: "best next opportunity" ≠ "biggest market today". The growth–scale
map makes each candidate's position explicit — bubble size is the composite
score, and the dashed lines split the field into rising stars, heavyweights,
niche rockets and the slow lane.

![Growth-scale map](../charts/07_growth_scale_map.png)

Because weights are a stated choice, every raw component is published in
§4 — re-weight them to your own view and the ranking re-computes in one
run.

---

## 11. Limitations & Honest Notes

- **Candidate-pool selection is a judgment call.** Eight challengers were
  chosen for credible DS/ML momentum; a different pool yields a different
  ranking. The pool is config-driven and trivially extensible.
- **Momentum-weighting favours small bases.** A 30% CAGR on a $4B market is
  easier than on a $60B market; absolute-scale weightings would lift the
  heavyweights (Finance, Healthcare) relative to niche rockets.
- **Weight sensitivity.** Scores respond to the component weights; the raw
  components are published precisely so a reader can re-weight.
- **Data provenance.** The bundled dataset is *calibrated synthetic data* —
  realistic magnitudes matching public benchmarks, not scraped primary
  sources. Swapping in a real panel (`--from-csv`) is the single
  highest-value improvement.
- **Short series.** ~11 annual points per series limits model complexity;
  the two-model ensemble was chosen for parsimony and interpretability
  over SARIMA/Prophet-style flexibility.
- **Approximate uncertainty.** Bootstrap intervals assume iid innovations
  and treat ensemble weights as fixed; treat them as indicative, not exact.
- **Fixed adoption ceiling.** The logistic model assumes each industry has
  a stationary saturation level; a step-change (e.g. a regulatory shift)
  would invalidate the projection.
- **No shock modelling.** Geopolitical, regulatory or macro shocks
  (e.g. an AI act, a defence-spending surge) are outside the generating
  process of the historical window.
- **Sector overlap.** AI Infrastructure *supplies* the other sectors;
  cross-sector double counting is possible in market-size figures.

## 12. Reproduce / Extend

```bash
pip install -r requirements.txt
python main.py                      # full pipeline (data → report)
python main.py --from-csv data/industry_panel.csv   # use your own data
python main.py --horizon 8 --seed 7                 # different horizon/seed
```

## 13. Calibration References

- Grand View Research — *AI in Healthcare Market* (~$20B+, 2025 est.)
- Statista / Precedence — *AI market by industry* (finance, energy, cyber,
  retail, logistics, insurance, education)
- McKinsey — *The State of AI* surveys (adoption by industry/function)
- US Bureau of Labor Statistics — *Data Scientists* occupational outlook
- Gardner & McKenzie (1985) — damped-trend exponential smoothing
"""
    outpath.parent.mkdir(parents=True, exist_ok=True)
    outpath.write_text(md, encoding="utf-8")
