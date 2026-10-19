"""Ensemble time-series forecasting: exponential growth + damped Holt trend.

Design notes
------------
Only ~11 yearly observations exist per series, so heavyweight models
(SARIMA, Prophet, gradient boosting) would overfit. Instead two
interpretable models are ensembled:

1. ExponentialGrowth - OLS on log(y) ~ t, i.e. y_t = a * (1+g)^t.
   Captures compound growth, the dominant pattern for emerging tech markets.
2. DampedHolt - Holt's linear trend method with Gardner-McKenzie damping.
   Captures recent momentum while tempering long-horizon extrapolation.

Ensemble weights are learned by backtesting on a 3-year holdout, weighting
each model by inverse MAPE. Uncertainty is a *mixture bootstrap*: n paths
are simulated, each path drawn from a component model with probability
equal to its weight; 5th/95th percentiles give a 90% interval that reflects
both model disagreement and sampling noise.

Adoption rates are bounded [0, 100] and follow S-curves, so they are
handled separately with a closed-form logistic fit (grid search over the
ceiling, OLS on the logit for steepness/midpoint).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class ForecastResult:
    """Container for one (industry, metric) ensemble forecast."""

    industry: str
    metric: str
    years: list[int]
    point: np.ndarray
    lo: np.ndarray
    hi: np.ndarray
    hist_years: np.ndarray
    hist_values: np.ndarray
    horizon_cagr: float
    weights: dict[str, float]
    backtest_mape: dict[str, float]

    def __post_init__(self) -> None:
        # Keep the point forecast inside the simulated band.
        self.lo = np.minimum(self.lo, self.point)
        self.hi = np.maximum(self.hi, self.point)


# --------------------------------------------------------------------------
# Component models
# --------------------------------------------------------------------------
class ExponentialGrowth:
    """Compound-growth model: log(y) = intercept + slope * t (OLS)."""

    name = "ExponentialGrowth"

    def __init__(self) -> None:
        self.n = 0
        self.slope = 0.0
        self.intercept = 0.0
        self.growth = 0.0
        self.resid = np.array([])

    def fit(self, y: np.ndarray) -> "ExponentialGrowth":
        y = np.asarray(y, float)
        self.n = len(y)
        t = np.arange(self.n, dtype=float)
        self.slope, self.intercept = np.polyfit(t, np.log(y), 1)
        self.growth = float(np.exp(self.slope) - 1.0)
        self.resid = np.log(y) - (self.intercept + self.slope * t)
        return self

    def predict(self, h: int) -> np.ndarray:
        t = np.arange(self.n, self.n + h, dtype=float)
        return np.exp(self.intercept + self.slope * t)

    def simulate(self, h: int, n: int, rng: np.random.Generator) -> np.ndarray:
        t = np.arange(self.n, self.n + h, dtype=float)
        draws = rng.choice(self.resid, size=(n, h))
        return np.exp(self.intercept + self.slope * t + draws)


class DampedHolt:
    """Holt's linear trend with multiplicative damping (ETS-ish, grid-tuned)."""

    name = "DampedHolt"

    _PHIS = (0.80, 0.90, 0.95, 1.00)

    def __init__(self) -> None:
        self.alpha = 0.3
        self.beta = 0.1
        self.phi = 0.95
        self.level = 0.0
        self.trend = 0.0
        self.resid = np.array([])

    # -- internals ---------------------------------------------------------
    @staticmethod
    def _run(y: np.ndarray, alpha: float, beta: float, phi: float):
        n = len(y)
        lev = np.empty(n)
        tr = np.empty(n)
        res = np.zeros(n)
        lev[0] = y[0]
        tr[0] = y[1] - y[0] if n > 1 else 0.0
        for i in range(1, n):
            f = lev[i - 1] + phi * tr[i - 1]
            lev[i] = alpha * y[i] + (1.0 - alpha) * f
            tr[i] = beta * (lev[i] - lev[i - 1]) + (1.0 - beta) * phi * tr[i - 1]
            res[i] = y[i] - f
        return lev, tr, res

    # -- API ----------------------------------------------------------------
    def fit(self, y: np.ndarray) -> "DampedHolt":
        y = np.asarray(y, float)
        best = (np.inf, self.alpha, self.beta, self.phi)
        for alpha in np.arange(0.10, 1.00, 0.10):
            for beta in np.arange(0.05, 0.61, 0.05):
                for phi in self._PHIS:
                    _, _, res = self._run(y, alpha, beta, phi)
                    sse = float((res**2).sum())
                    if sse < best[0]:
                        best = (sse, alpha, beta, phi)
        self.alpha, self.beta, self.phi = best[1], best[2], best[3]
        lev, tr, res = self._run(y, self.alpha, self.beta, self.phi)
        self.level = float(lev[-1])
        self.trend = float(tr[-1])
        self.resid = res[1:]  # first residual is a placeholder
        return self

    def predict(self, h: int) -> np.ndarray:
        phis = self.phi ** np.arange(1, h + 1)
        return self.level + self.trend * np.cumsum(phis)

    def simulate(self, h: int, n: int, rng: np.random.Generator) -> np.ndarray:
        draws = rng.choice(self.resid, size=(n, h))
        lev = np.full(n, self.level)
        tr = np.full(n, self.trend)
        paths = np.empty((n, h))
        for s in range(h):
            f = lev + self.phi * tr
            paths[:, s] = f + draws[:, s]
            lev = f + self.alpha * draws[:, s]
            tr = self.phi * tr + self.beta * draws[:, s]
        return np.clip(paths, 0.05, None)


# --------------------------------------------------------------------------
# Ensemble
# --------------------------------------------------------------------------
def _mape(actual: np.ndarray, pred: np.ndarray) -> float:
    return float(np.mean(np.abs(actual - pred) / actual))


def ensemble_forecast(
    industry: str,
    metric: str,
    years: np.ndarray,
    values: np.ndarray,
    horizon: int = 5,
    n_boot: int = 1000,
    holdout: int = 3,
    seed: int = 0,
) -> ForecastResult:
    """Backtested ensemble forecast with mixture-bootstrap 90% CI."""
    years = np.asarray(years, int)
    y = np.asarray(values, float)
    rng = np.random.default_rng(seed)

    # 1) Backtest on a holdout to learn model weights.
    use_backtest = len(y) >= 8 and holdout >= 1
    if use_backtest:
        train, test = y[:-holdout], y[-holdout:]
        m = ExponentialGrowth().fit(train)
        h = DampedHolt().fit(train)
        mape = {
            ExponentialGrowth.name: _mape(test, m.predict(holdout)),
            DampedHolt.name: _mape(test, h.predict(holdout)),
        }
        inv = {k: 1.0 / (v + 1e-3) for k, v in mape.items()}
        total = sum(inv.values())
        weights = {k: v / total for k, v in inv.items()}
    else:
        mape = {"backtest": float("nan")}
        weights = {ExponentialGrowth.name: 0.5, DampedHolt.name: 0.5}

    # 2) Refit both models on the full history.
    full = [ExponentialGrowth().fit(y), DampedHolt().fit(y)]

    # 3) Point forecast = weight-weighted average of component forecasts.
    points = [m.predict(horizon) for m in full]
    w_vec = np.array([weights[m.name] for m in full])
    point = np.average(np.vstack(points), axis=0, weights=w_vec)

    # 4) Mixture bootstrap for the 90% interval.
    sims = [m.simulate(horizon, n_boot, rng) for m in full]
    idx = rng.choice(len(full), size=n_boot, p=w_vec)
    mixture = np.where(
        (idx == 0)[:, None], sims[0], sims[1]
    )
    lo, hi = np.percentile(mixture, [5.0, 95.0], axis=0)

    horizon_cagr = float((point[-1] / y[-1]) ** (1.0 / horizon) - 1.0)
    return ForecastResult(
        industry=industry,
        metric=metric,
        years=list(range(int(years[-1]) + 1, int(years[-1]) + 1 + horizon)),
        point=point,
        lo=lo,
        hi=hi,
        hist_years=years,
        hist_values=y,
        horizon_cagr=horizon_cagr,
        weights=weights,
        backtest_mape=mape,
    )


# --------------------------------------------------------------------------
# Logistic adoption model (bounded metric)
# --------------------------------------------------------------------------
def fit_logistic(years: np.ndarray, y: np.ndarray):
    """Closed-form-ish logistic fit: grid over ceiling L, OLS on logit.

    Returns (ceiling, k, midpoint, sse).
    """
    t = np.asarray(years, float)
    y = np.asarray(y, float)
    best = None
    for L in np.linspace(y.max() * 1.02, 99.0, 50):
        z = np.clip(y / L, 1e-4, 1.0 - 1e-4)
        x = np.log(z / (1.0 - z))
        k, b = np.polyfit(t, x, 1)
        pred = L / (1.0 + np.exp(-(k * t + b)))
        sse = float(((y - pred) ** 2).sum())
        if best is None or sse < best[0]:
            best = (sse, L, k, b)
    sse, L, k, b = best
    t0 = -b / k if k != 0 else float(t.mean())
    return float(L), float(k), float(t0), sse


def logistic_value(ceiling: float, k: float, midpoint: float, year: float) -> float:
    return float(ceiling / (1.0 + np.exp(-k * (year - midpoint))))
