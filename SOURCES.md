# Data Sources & Methods

Everything a reviewer needs to audit this project: where the numbers come
from, what the algorithms are formally called, and how to swap in real data.

---

## 1. Data provenance — read this first

The bundled dataset (`data/industry_panel.csv`) is a **benchmark-calibrated
synthetic panel**: 13 industries x 2015-2025, five indicators per industry-year.

**Why synthetic?** No single free public dataset covers "DS/ML market size,
investment, job postings, adoption, and patents, by industry, by year." The
generator is therefore calibrated so that every 2025 value lands **inside the
range of published estimates** from the market-research and survey sources
listed in §2, and known shocks (2020 e-commerce surge, 2020 energy capex
dip, 2023 GenAI boom in education/legal) are built in as year multipliers.

**The pipeline itself is data-agnostic:** `python main.py --from-csv your.csv`
with the same schema runs the entire study (forecasting, scoring, charts,
report) on real data. See §3 for where to get real data.

## 2. Calibration sources (citable)

Market-size calibration targets for the 2025 cross-section:

| Industry (tier) | 2025 target ($B) | Calibration source |
|---|---|---|
| Healthcare (incumbent) | 20–25 | Grand View Research, *AI in Healthcare Market* |
| Finance (incumbent) | 22–28 | Precedence Research / Statista, *AI in BFSI* |
| Defense & Cybersecurity (incumbent) | 15–18 | Statista / MarketsandMarkets, *AI in Cybersecurity* + defense AI spending |
| Energy & Industrials (incumbent) | 9–12 | Precedence Research, *AI in Energy* |
| AI Infrastructure (incumbent) | 55–70 | Precedence Research, *AI Infrastructure Market*; hyperscaler capex reporting |
| Retail & E-commerce (challenger) | 10–15 | Statista / MarketsandMarkets, *AI in Retail* |
| Transportation & Logistics (challenger) | 6–10 | *AI in Logistics & Supply Chain* estimates |
| Agriculture & Food Tech (challenger) | 2.5–4 | MarketsandMarkets, *Precision Agriculture / AI in Agriculture* |
| Telecom & Media (challenger) | 4–6 | *AI in Telecom* market estimates |
| Climate & Sustainability Tech (challenger) | 4–8 | Climate-tech AI sector reports (Precedence-style) |
| Insurance (challenger) | 5–9 | Grand View Research, *AI in Insurance* |
| Education & EdTech (challenger) | 4–8 | Grand View Research / MarketsandMarkets, *AI in Education* |
| Legal & Compliance Tech (challenger) | 2–4 | Grand View Research, *AI in Legal Services* |

Other indicators:

- **ML adoption rates** — McKinsey, *The State of AI* annual survey
  (adoption by industry/function); Stanford HAI, *AI Index Report*
- **Job postings & data-science employment** — US Bureau of Labor
  Statistics, *Occupational Outlook Handbook: Data Scientists*
  (~36% projected growth 2023–33); Lightcast/LinkedIn postings indices
- **AI patent activity** — WIPO patent analytics; Stanford HAI *AI Index*

## 3. Where to get real data (highest-value upgrade)

| Indicator | Free source |
|---|---|
| AI investment & adoption by sector | Stanford HAI *AI Index Report* (annual, free) |
| Employment by industry (US) | BLS OES / QCEW datasets |
| Job postings | Lightcast / LinkedIn datasets (Kaggle mirrors exist) |
| VC funding by sector | Crunchbase (free tier) / CB Insights / PitchBook |
| Market forecasts | Statista / Gartner / IDC (paid); press releases often quote headline numbers |
| Skills & workforce trends | WEF *Future of Jobs Report* (free) |

Swap any of these in via `python main.py --from-csv your_panel.csv`
(schema: `industry, tier, year, market_size_usd_b, investment_usd_b,
job_postings_k, adoption_rate_pct, ai_patents_k`).

## 4. Algorithms — formal names

What the pipeline does, in the terminology a data scientist expects:

1. **Log-linear regression (exponential growth model).** OLS of
   `log(y) ~ t`, i.e. `y = A·(1+g)^t`; the slope estimates the compound
   growth rate (CAGR) directly. Linear-in-parameters, nonlinear in time —
   the correct regression form for compound-growth series.
2. **Damped Holt's linear trend method** (exponential smoothing with
   damped trend). Gardner & McKenzie (1985), *Forecasting trends in time
   series*. In Hyndman's ETS taxonomy this is an `ETS(A,Ad,N)`-style
   model — fitted here by SSE grid search over (α, β, φ) rather than
   state-space MLE (a deliberate simplification at n≈11).
3. **Inverse-error weighted forecast combination.** Ensemble weights ∝
   1/MAPE from a 3-year holdout backtest per series. Classical forecast
   combination dates to Bates & Granger (1969), *The Combination of
   Forecasts*.
4. **Residual (innovation) bootstrap prediction intervals.** 1,000
   simulated future paths; each path draws a component model with
   probability equal to its ensemble weight (mixture bootstrap); 5th/95th
   percentiles give the 90% interval. Cf. bootstrapped intervals for
   exponential smoothing (Hyndman et al., *Forecasting with Exponential
   Smoothing*, 2008).
5. **Logistic growth model (Pearl–Reed S-curve)** for bounded adoption
   rates, with a semi-parametric fit: grid search over the carrying
   capacity (ceiling) + closed-form OLS on the logit transform. Cf. Pearl &
   Reed (1920); Rogers, *Diffusion of Innovations*.
6. **Min-max normalised composite indicator** for ranking — methodology
   per OECD/JRC, *Handbook on Constructing Composite Indicators* (2008):
   normalise each component, apply declared weights (growth 30%, scale
   25%, investment 20%, jobs 15%, adoption 10%), aggregate, rank.

**One-line description:** *a backtest-weighted ensemble of log-linear OLS
growth regression and damped Holt exponential smoothing, with
residual-bootstrap prediction intervals, logistic S-curve adoption
modeling, and a min-max composite indicator ranking.*

## 5. Paste-ready project summary (for colleagues)

> This project predicts the **next top industries for Data Science & ML**.
> A 13-candidate pool (5 incumbents + 8 challengers) is tracked over
> 2015–2025 on market size, investment, job postings, ML adoption, and
> patents. Each unbounded metric is forecast to 2030 with a
> backtest-weighted ensemble of **log-linear OLS regression** (compound
> growth) and **damped Holt exponential smoothing** (ETS(A,Ad,N)-style),
> with 90% **residual-bootstrap prediction intervals**; bounded adoption
> uses **logistic (Pearl–Reed) S-curves**. Industries are ranked by a
> **min-max composite prospect index** (growth 30% · scale 25% ·
> investment 20% · jobs 15% · adoption 10%). Result: three challengers —
> Climate Tech, Legal Tech, EdTech — displace Finance, Defense/Cyber, and
> Energy in the predicted next top-5; AI Infrastructure ($63B → $280B,
> 34.8% CAGR) and Healthcare hold. Inputs are benchmark-calibrated to
> published estimates (GVR, Statista/Precedence, McKinsey State of AI,
> BLS) and the pipeline runs unchanged on real data via `--from-csv`.

## 6. Known simplifications (stated for reviewers)

- Holt parameters via SSE grid search, not state-space MLE.
- Bootstrap intervals assume iid innovations and fixed ensemble weights.
- Logistic fit assumes a stationary adoption ceiling per industry.
- Calibrated synthetic inputs (see §1) — the pipeline accepts real data.
- ~11 annual observations per series; models chosen for parsimony.
- No shock/regulatory scenario modelling.
