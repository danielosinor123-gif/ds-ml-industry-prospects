"""Composite prospect index: ranks industries by their 2026-2030 DS/ML outlook.

score(industry) = sum_c  w_c * minmax_c(raw_c(industry)) * 100

Components (weights reflect what makes an industry a *good prospect*
for DS/ML professionals and investors):

    growth_momentum      forecast market-size CAGR, 2025 -> 2030   30%
    market_scale         forecast 2030 market size ($B)            25%
    investment_momentum forecast investment CAGR, 2025 -> 2030     20%
    job_demand          forecast job-postings CAGR                 15%
    ai_maturity         projected 2030 ML adoption rate            10%

Min-max normalisation keeps every component on a comparable 0-1 scale
across the five industries.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

COMPONENT_WEIGHTS: dict[str, float] = {
    "growth_momentum": 0.30,
    "market_scale": 0.25,
    "investment_momentum": 0.20,
    "job_demand": 0.15,
    "ai_maturity": 0.10,
}

COMPONENT_LABELS: dict[str, str] = {
    "growth_momentum": "Growth momentum (market CAGR)",
    "market_scale": "Market scale (2030 $B)",
    "investment_momentum": "Investment momentum",
    "job_demand": "Job demand growth",
    "ai_maturity": "AI maturity (2030 adoption)",
}


def _minmax(values: np.ndarray) -> np.ndarray:
    mn, mx = values.min(), values.max()
    if mx - mn < 1e-12:
        return np.full(len(values), 0.5)
    return (values - mn) / (mx - mn)


def build_score_table(
    raw: dict[str, dict[str, float]], tiers: dict[str, str] | None = None
) -> pd.DataFrame:
    """Turn per-industry component values into a ranked score table.

    raw    = {industry: {component_name: value, ...}, ...}
    tiers  = optional {industry: "Incumbent" | "Challenger", ...}
    """
    industries = list(raw.keys())
    components = list(COMPONENT_WEIGHTS.keys())

    raw_df = pd.DataFrame(
        {ind: [raw[ind][c] for c in components] for ind in industries},
        index=components,
    ).T
    raw_df.index.name = "industry"

    norm = pd.DataFrame(index=raw_df.index)
    for comp in components:
        norm[comp] = _minmax(raw_df[comp].to_numpy())

    weights = np.array([COMPONENT_WEIGHTS[c] for c in components])
    scores = (norm[components].to_numpy() * weights).sum(axis=1) * 100.0

    out = pd.DataFrame(index=raw_df.index)
    out["score"] = np.round(scores, 1)
    for comp in components:
        out[comp] = raw_df[comp]
        out[f"{comp}_norm"] = np.round(norm[comp].to_numpy(), 3)
    out = out.sort_values("score", ascending=False)
    out.insert(0, "rank", range(1, len(out) + 1))
    if tiers is not None:
        out.insert(1, "tier", [str(tiers.get(ind, "")) for ind in out.index])
    out.index.name = "industry"
    return out


def strongest_component(row: pd.Series) -> str:
    """Name of the component where an industry is relatively strongest."""
    norm_cols = [f"{c}_norm" for c in COMPONENT_WEIGHTS]
    vals = {c: row[f"{c}_norm"] for c in COMPONENT_WEIGHTS}
    return max(vals, key=vals.get)
