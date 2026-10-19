"""Benchmark-calibrated dataset builder for the industry DS/ML prospect study.

Generates a panel dataset (13 candidate industries x 2015-2025) describing
Data Science & Machine Learning activity per industry:

    market_size_usd_b   Annual spending on DS/ML within the industry ($B)
    investment_usd_b    Funding & R&D capital directed at DS/ML ($B)
    job_postings_k      DS/ML job postings per year (thousands)
    adoption_rate_pct   Share of firms deploying ML in production (%)
    ai_patents_k        ML-related patents granted per year (thousands)

The candidate pool has two tiers:

    Incumbents  - today's recognised top-5 industries for DS/ML
                  (the industries from the original research list)
    Challengers - eight rising industries plausibly contesting the next
                  top spots, calibrated to published estimates

Incumbent calibration targets (public benchmarks, 2025 estimates):
    Healthcare               $20-25B market  (Grand View Research, "~$20B+")
    Finance                  $22-28B market  (~$25B, banking + fintech AI)
    Defense & Cybersecurity  $15-18B market  (~$16-20B, AI in cyber/defense)
    Energy & Industrials     $9-12B market   (~$10-12B, grid + predictive Mx)
    AI Infrastructure        $55-70B market  (~$50-100B, compute + MLOps)

Challenger calibration targets (public benchmarks, 2025 estimates):
    Retail & E-commerce           ~$11B  (recommendation, pricing, demand AI)
    Transportation & Logistics   ~$7B   (routing, warehouse robotics)
    Agriculture & Food Tech       ~$3.6B (precision farming, yield models)
    Telecom & Media               ~$4.4B (network optimisation, recsys)
    Climate & Sustainability Tech ~$4.5B (emissions, grid, climate risk)
    Insurance                     ~$6B   (claims, underwriting, fraud)
    Education & EdTech            ~$4.8B (adaptive learning, LLM tutors)
    Legal & Compliance Tech       ~$3.2B (contract AI, e-discovery)

HONESTY NOTE (important for a portfolio): these values are *calibrated
estimates*, not scraped primary data. The pipeline is fully data-agnostic:
run `python main.py --from-csv my_panel.csv` with the same schema and every
downstream step (forecasting, scoring, charts, report) runs unchanged.
Adding another candidate industry is a single new entry in
INDUSTRY_CALIBRATION below.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

HISTORY_YEARS = list(range(2015, 2026))
SEED = 42

TIER_INCUMBENT = "Incumbent"
TIER_CHALLENGER = "Challenger"

INCUMBENT_NAMES = {
    "Healthcare",
    "Finance",
    "Defense & Cybersecurity",
    "Energy & Industrials",
    "AI Infrastructure",
}

# Per-industry calibration parameters. "events" are year-specific
# multipliers capturing known shocks (e.g. the 2020 energy capex dip, the
# 2020 e-commerce surge, the 2023 GenAI boom in education/legal tech).
INDUSTRY_CALIBRATION: dict[str, dict] = {
    # ------------------------------------------------------- incumbents
    "Healthcare": {
        "tier": TIER_INCUMBENT,
        "base_market": 1.2, "market_cagr": 0.33,
        "invest_alpha": 0.55, "invest_extra": 0.015,
        "base_jobs": 6.0, "jobs_cagr": 0.22,
        "adopt_ceiling": 78.0, "adopt_mid": 2021.6, "adopt_k": 0.55,
        "base_patents": 0.8, "patents_cagr": 0.26,
        "events": {2020: 1.06},
    },
    "Finance": {
        "tier": TIER_INCUMBENT,
        "base_market": 3.0, "market_cagr": 0.235,
        "invest_alpha": 0.45, "invest_extra": 0.010,
        "base_jobs": 14.0, "jobs_cagr": 0.14,
        "adopt_ceiling": 82.0, "adopt_mid": 2018.8, "adopt_k": 0.60,
        "base_patents": 1.1, "patents_cagr": 0.18,
        "events": {2020: 0.98},
    },
    "Defense & Cybersecurity": {
        "tier": TIER_INCUMBENT,
        "base_market": 2.0, "market_cagr": 0.235,
        "invest_alpha": 0.60, "invest_extra": 0.010,
        "base_jobs": 7.0, "jobs_cagr": 0.19,
        "adopt_ceiling": 80.0, "adopt_mid": 2019.8, "adopt_k": 0.55,
        "base_patents": 0.9, "patents_cagr": 0.22,
        "events": {2020: 1.03},
    },
    "Energy & Industrials": {
        "tier": TIER_INCUMBENT,
        "base_market": 1.4, "market_cagr": 0.225,
        "invest_alpha": 0.42, "invest_extra": 0.005,
        "base_jobs": 5.5, "jobs_cagr": 0.20,
        "adopt_ceiling": 72.0, "adopt_mid": 2022.0, "adopt_k": 0.50,
        "base_patents": 0.7, "patents_cagr": 0.20,
        "events": {2020: 0.93, 2021: 0.97},
    },
    "AI Infrastructure": {
        "tier": TIER_INCUMBENT,
        "base_market": 3.0, "market_cagr": 0.36,
        "invest_alpha": 0.85, "invest_extra": 0.020,
        "base_jobs": 5.0, "jobs_cagr": 0.24,
        "adopt_ceiling": 90.0, "adopt_mid": 2018.5, "adopt_k": 0.65,
        "base_patents": 1.5, "patents_cagr": 0.30,
        "events": {2020: 1.08},
    },
    # ------------------------------------------------------- challengers
    "Retail & E-commerce": {
        "tier": TIER_CHALLENGER,
        "base_market": 1.1, "market_cagr": 0.26,
        "invest_alpha": 0.50, "invest_extra": 0.010,
        "base_jobs": 5.5, "jobs_cagr": 0.20,
        "adopt_ceiling": 74.0, "adopt_mid": 2020.5, "adopt_k": 0.50,
        "base_patents": 0.6, "patents_cagr": 0.22,
        "events": {2020: 1.12},
    },
    "Transportation & Logistics": {
        "tier": TIER_CHALLENGER,
        "base_market": 0.8, "market_cagr": 0.25,
        "invest_alpha": 0.50, "invest_extra": 0.015,
        "base_jobs": 4.0, "jobs_cagr": 0.21,
        "adopt_ceiling": 70.0, "adopt_mid": 2021.3, "adopt_k": 0.48,
        "base_patents": 0.5, "patents_cagr": 0.24,
        "events": {2020: 0.97},
    },
    "Agriculture & Food Tech": {
        "tier": TIER_CHALLENGER,
        "base_market": 0.30, "market_cagr": 0.28,
        "invest_alpha": 0.55, "invest_extra": 0.020,
        "base_jobs": 1.5, "jobs_cagr": 0.23,
        "adopt_ceiling": 62.0, "adopt_mid": 2022.5, "adopt_k": 0.45,
        "base_patents": 0.4, "patents_cagr": 0.25,
        "events": {},
    },
    "Telecom & Media": {
        "tier": TIER_CHALLENGER,
        "base_market": 0.40, "market_cagr": 0.27,
        "invest_alpha": 0.45, "invest_extra": 0.005,
        "base_jobs": 3.0, "jobs_cagr": 0.18,
        "adopt_ceiling": 76.0, "adopt_mid": 2020.0, "adopt_k": 0.50,
        "base_patents": 0.5, "patents_cagr": 0.22,
        "events": {2020: 1.05},
    },
    "Climate & Sustainability Tech": {
        "tier": TIER_CHALLENGER,
        "base_market": 0.25, "market_cagr": 0.34,
        "invest_alpha": 0.70, "invest_extra": 0.025,
        "base_jobs": 1.2, "jobs_cagr": 0.28,
        "adopt_ceiling": 68.0, "adopt_mid": 2022.2, "adopt_k": 0.55,
        "base_patents": 0.3, "patents_cagr": 0.32,
        "events": {2021: 1.08},
    },
    "Insurance": {
        "tier": TIER_CHALLENGER,
        "base_market": 0.70, "market_cagr": 0.24,
        "invest_alpha": 0.48, "invest_extra": 0.010,
        "base_jobs": 3.5, "jobs_cagr": 0.16,
        "adopt_ceiling": 78.0, "adopt_mid": 2019.5, "adopt_k": 0.55,
        "base_patents": 0.5, "patents_cagr": 0.20,
        "events": {2020: 0.99},
    },
    "Education & EdTech": {
        "tier": TIER_CHALLENGER,
        "base_market": 0.35, "market_cagr": 0.30,
        "invest_alpha": 0.60, "invest_extra": 0.020,
        "base_jobs": 1.8, "jobs_cagr": 0.24,
        "adopt_ceiling": 72.0, "adopt_mid": 2021.8, "adopt_k": 0.50,
        "base_patents": 0.4, "patents_cagr": 0.28,
        "events": {2020: 1.15, 2023: 1.22},
    },
    "Legal & Compliance Tech": {
        "tier": TIER_CHALLENGER,
        "base_market": 0.20, "market_cagr": 0.32,
        "invest_alpha": 0.65, "invest_extra": 0.025,
        "base_jobs": 1.0, "jobs_cagr": 0.26,
        "adopt_ceiling": 66.0, "adopt_mid": 2022.8, "adopt_k": 0.60,
        "base_patents": 0.25, "patents_cagr": 0.30,
        "events": {2023: 1.28},
    },
}

PANEL_COLUMNS = [
    "industry", "tier", "year", "market_size_usd_b", "investment_usd_b",
    "job_postings_k", "adoption_rate_pct", "ai_patents_k",
]


def build_panel(seed: int = SEED) -> pd.DataFrame:
    """Generate the industry x year panel with reproducible noise."""
    rng = np.random.default_rng(seed)
    rows: list[dict] = []

    for industry, cfg in INDUSTRY_CALIBRATION.items():
        for i, year in enumerate(HISTORY_YEARS):
            event = cfg["events"].get(year, 1.0)

            market = (
                cfg["base_market"]
                * (1.0 + cfg["market_cagr"]) ** i
                * event
                * np.exp(rng.normal(0.0, 0.045))
            )
            invest = (
                market
                * cfg["invest_alpha"]
                * (1.0 + cfg["invest_extra"]) ** i
                * np.exp(rng.normal(0.0, 0.060))
            )
            jobs = (
                cfg["base_jobs"]
                * (1.0 + cfg["jobs_cagr"]) ** i
                * event
                * np.exp(rng.normal(0.0, 0.050))
            )
            adoption = cfg["adopt_ceiling"] / (
                1.0 + np.exp(-cfg["adopt_k"] * (year - cfg["adopt_mid"]))
            )
            adoption = float(np.clip(adoption + rng.normal(0.0, 0.8), 1.0, 99.0))
            patents = (
                cfg["base_patents"]
                * (1.0 + cfg["patents_cagr"]) ** i
                * np.exp(rng.normal(0.0, 0.070))
            )

            rows.append(
                {
                    "industry": industry,
                    "tier": cfg["tier"],
                    "year": year,
                    "market_size_usd_b": round(float(market), 3),
                    "investment_usd_b": round(float(invest), 3),
                    "job_postings_k": round(float(jobs), 3),
                    "adoption_rate_pct": round(adoption, 2),
                    "ai_patents_k": round(float(patents), 3),
                }
            )

    return pd.DataFrame(rows, columns=PANEL_COLUMNS)


def load_panel_from_csv(path: str) -> pd.DataFrame:
    """Load an external panel with the same schema as build_panel() output.

    The 'tier' column is optional: if absent, industries in INCUMBENT_NAMES
    are labelled Incumbent and everything else Challenger.
    """
    df = pd.read_csv(path)
    missing = [c for c in PANEL_COLUMNS if c != "tier" and c not in df.columns]
    if missing:
        raise ValueError(
            f"CSV {path} is missing required columns: {missing}. "
            f"Expected schema: {PANEL_COLUMNS}"
        )
    if "tier" not in df.columns:
        df["tier"] = [
            TIER_INCUMBENT if name in INCUMBENT_NAMES else TIER_CHALLENGER
            for name in df["industry"]
        ]
    return df
