"""Mini-Project 5: Taxi Route Revenue, Pricing & Fleet Planner.

A matatu association runs 14-seater vehicles on three routes out of Kampala.
It wants to forecast demand, judge its fares and size its fleet.

Classes
-------
:class:`Route`
    One route's passenger counts and fare.
:class:`FareMarket`
    Linear supply and demand, solved as a 2x2 system for the clearing fare.
:class:`MovingAverage`, :class:`ExponentialSmoothing`, :class:`StraightLine`,
:class:`WeightedAverage`, :class:`SeasonalNaive`
    Forecasting models, all filling in the two hooks of
    :class:`~src.forecasting.Forecaster`.
:class:`FleetSizer`
    Turns a passenger forecast into a vehicle count.

Backtesting is **not** reimplemented here. ``WalkForwardMixin`` and
``grid_search`` already live in :mod:`src.mixins`, shared with mini-project 1,
so every model in this file can backtest and tune itself without a line of new
code. That reuse is the point of putting the capability in a mixin rather than
in a project module.
"""
from __future__ import annotations

import statistics
from dataclasses import dataclass
from typing import Callable, Iterator, Sequence

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike
from scipy import linalg

from src.forecasting import EnsembleForecaster, Forecaster
from src.metrics import MetricSuite
from src.mixins import SeriesValidationMixin, grid_search

__all__ = ["ROUTE_DATA", "ExponentialSmoothing", "FareMarket", "FleetSizer",
           "MovingAverage", "Route", "SeasonalNaive", "StraightLine",
           "WeightedAverage", "compare_models", "load_routes",
           "simulate_weekly_pattern", "tune_smoothing"]

#: Daily passenger counts over ten days, with the fare in UGX. The Ntinda
#: figures come from the brief; the other two are illustrative.
ROUTE_DATA: dict[str, dict[str, object]] = {
    "Kampala-Ntinda": {"counts": (35, 40, 42, 50, 55, 60, 48, 52, 47, 45),
                       "fare": 2_000.0},
    "Kampala-Entebbe": {"counts": (60, 58, 65, 70, 72, 80, 75, 68, 66, 64),
                        "fare": 5_000.0},
    "Kampala-Mukono": {"counts": (45, 47, 50, 49, 55, 62, 58, 53, 51, 50),
                       "fare": 3_000.0},
}


class Route(SeriesValidationMixin):
    """One route's daily passenger counts and its fare.

    Parameters
    ----------
    name : str
        Route name.
    counts : array-like of float
        Daily passenger counts, non-negative and finite.
    fare : float
        Fare per passenger in UGX, strictly positive.
    """

    min_obs = 1

    def __init__(self, name: str, counts: ArrayLike, fare: float) -> None:
        if not isinstance(name, str) or not name.strip():
            raise ValueError("name must be a non-empty string")
        values = self.clean(counts, "counts", self.min_obs)
        if np.any(values < 0):
            raise ValueError("passenger counts cannot be negative")
        if not np.isfinite(fare) or fare <= 0:
            raise ValueError("fare must be positive and finite")
        self.name = name.strip()
        self.counts = values
        self.fare = float(fare)

    def __repr__(self) -> str:
        return (f"<Route {self.name}: {len(self)} days, UGX {self.fare:,.0f}, "
                f"total UGX {self.total_revenue:,.0f}>")

    def __len__(self) -> int:
        return int(self.counts.size)

    def __getitem__(self, day: int) -> float:
        """``route[1]`` is day one, counting from 1 as the association does."""
        if not 1 <= day <= len(self):
            raise KeyError(f"day {day} is outside 1-{len(self)}")
        return float(self.counts[day - 1])

    def __iter__(self) -> Iterator[tuple[int, float]]:
        return enumerate(self.counts.tolist(), start=1)

    @property
    def revenue_by_day(self) -> np.ndarray:
        return self.counts * self.fare

    @property
    def total_revenue(self) -> float:
        return float(self.revenue_by_day.sum())

    def describe(self) -> dict[str, float]:
        """Mean, median, variance, sd and CV of passengers via ``statistics``."""
        data = self.counts.tolist()
        if len(data) < 2:
            raise ValueError("need at least two days for a variance")
        mean, sd = statistics.mean(data), statistics.stdev(data)
        return {"mean": mean, "median": statistics.median(data),
                "variance": statistics.variance(data), "stdev": sd,
                "cv": sd / mean if mean else float("inf")}


@dataclass(frozen=True)
class FareMarket:
    """Linear supply and demand for one route.

    With :math:`Q_d = a - bP` and :math:`Q_s = c + dP`, rewriting both as
    :math:`Q + bP = a` and :math:`Q - dP = c` gives a 2x2 system in
    :math:`(P, Q)` that ``scipy.linalg.solve`` handles directly.

    Parameters
    ----------
    demand_at_zero, demand_slope : float
        ``a`` and ``b``; the brief gives 120 and 0.02.
    supply_at_zero, supply_slope : float
        ``c`` and ``d``; the brief gives 10 and 0.03.
    """

    demand_at_zero: float = 120.0
    demand_slope: float = 0.02
    supply_at_zero: float = 10.0
    supply_slope: float = 0.03

    def __post_init__(self) -> None:
        if self.demand_slope <= 0 or self.supply_slope <= 0:
            raise ValueError("demand must fall and supply must rise with price")
        if self.demand_at_zero <= self.supply_at_zero:
            raise ValueError("with demand below supply at zero price there is no market")

    def as_system(self) -> tuple[np.ndarray, np.ndarray]:
        """Coefficient matrix and right-hand side, unknowns ordered (P, Q)."""
        A = np.array([[self.demand_slope, 1.0], [-self.supply_slope, 1.0]])
        return A, np.array([self.demand_at_zero, self.supply_at_zero])

    @property
    def det(self) -> float:
        return float(np.linalg.det(self.as_system()[0]))

    @property
    def cond(self) -> float:
        return float(np.linalg.cond(self.as_system()[0]))

    def clearing(self) -> tuple[float, float]:
        """The market-clearing ``(fare, quantity)``."""
        if np.isclose(self.det, 0.0):
            raise ValueError("supply and demand are parallel; nothing clears")
        A, b = self.as_system()
        price, quantity = linalg.solve(A, b)
        return float(price), float(quantity)

    def demanded(self, price: float) -> float:
        return self.demand_at_zero - self.demand_slope * price

    def supplied(self, price: float) -> float:
        return self.supply_at_zero + self.supply_slope * price

    def imbalance(self, price: float) -> dict[str, float]:
        """Excess demand (shortage) or excess supply at a given fare."""
        if price < 0:
            raise ValueError("price cannot be negative")
        clearing_price, _ = self.clearing()
        wanted, offered = self.demanded(price), self.supplied(price)
        return {"price": price, "demanded": wanted, "supplied": offered,
                "shortage": wanted - offered,
                "below_clearing_by": clearing_price - price}


# ----------------------------------------------------------------------
# Task 3: the models
# ----------------------------------------------------------------------
class MovingAverage(Forecaster):
    """Mean of the last ``window`` days, projected flat.

    The association's current method. Robust to one odd day, but blind to any
    trend and lagging whatever it follows by roughly half the window.
    """

    label = "moving-average"

    def __init__(self, window: int = 3) -> None:
        super().__init__()
        self.window = self.whole_number(window, "window")
        self.min_obs = self.window
        self.level: float | None = None

    def _estimate(self) -> None:
        self.level = float(np.mean(self.y_[-self.window:]))

    def _extrapolate(self, horizon: int) -> np.ndarray:
        return np.full(horizon, self.level)

    def fitted(self) -> np.ndarray:
        if self.y_ is None:
            raise RuntimeError("call fit() before fitted() on MovingAverage")
        out = np.full_like(self.y_, np.nan)
        for i in range(self.window, self.y_.size):
            out[i] = self.y_[i - self.window:i].mean()
        return out


class WeightedAverage(Forecaster):
    """Moving average with linearly increasing weights.

    A compromise between the flat moving average and the naive forecast: the
    most recent day counts most, but older days still damp the noise.
    """

    label = "weighted-average"

    def __init__(self, window: int = 3) -> None:
        super().__init__()
        self.window = self.whole_number(window, "window")
        self.min_obs = self.window
        self.weights = np.arange(1, self.window + 1, dtype=float)
        self.weights /= self.weights.sum()
        self.level: float | None = None

    def _estimate(self) -> None:
        self.level = float(np.dot(self.y_[-self.window:], self.weights))

    def _extrapolate(self, horizon: int) -> np.ndarray:
        return np.full(horizon, self.level)


class ExponentialSmoothing(Forecaster):
    """Simple exponential smoothing with a tunable ``alpha``.

    The level updates as ``l_t = alpha*y_t + (1-alpha)*l_{t-1}``, so ``alpha``
    controls how fast the model forgets. At ``alpha = 1`` it is the naive
    forecast; near zero it is a long-run average.
    """

    min_obs = 2

    def __init__(self, alpha: float = 0.5) -> None:
        super().__init__()
        if not 0 < alpha <= 1:
            raise ValueError("alpha must be in (0, 1]")
        self.alpha = float(alpha)
        self.label = f"ses({self.alpha:.2f})"
        self.levels: np.ndarray | None = None

    def _estimate(self) -> None:
        levels = np.empty_like(self.y_)
        levels[0] = self.y_[0]
        for i in range(1, self.y_.size):
            levels[i] = self.alpha * self.y_[i] + (1 - self.alpha) * levels[i - 1]
        self.levels = levels

    def _extrapolate(self, horizon: int) -> np.ndarray:
        return np.full(horizon, float(self.levels[-1]))

    def fitted(self) -> np.ndarray:
        if self.y_ is None:
            raise RuntimeError("call fit() before fitted() on ExponentialSmoothing")
        out = np.full_like(self.y_, np.nan)
        out[1:] = self.levels[:-1]
        return out


class StraightLine(Forecaster):
    """Least-squares linear trend, fitted with ``np.polyfit`` at degree 1."""

    label = "straight-line"
    min_obs = 2

    def __init__(self) -> None:
        super().__init__()
        self.gradient: float | None = None
        self.offset: float | None = None

    def _estimate(self) -> None:
        self.gradient, self.offset = (
            float(c) for c in np.polyfit(self.t_, self.y_, 1))

    def _extrapolate(self, horizon: int) -> np.ndarray:
        ahead = self.t_[-1] + np.arange(1, horizon + 1, dtype=float)
        return self.gradient * ahead + self.offset

    def fitted(self) -> np.ndarray:
        if self.y_ is None:
            raise RuntimeError("call fit() before fitted() on StraightLine")
        return self.gradient * self.t_ + self.offset


class SeasonalNaive(Forecaster):
    """Repeats the value from ``period`` steps ago.

    Useless on ten days, and the obvious model once a weekly rhythm exists --
    which the extension demonstrates.
    """

    label = "seasonal-naive"

    def __init__(self, period: int = 7) -> None:
        super().__init__()
        self.period = self.whole_number(period, "period")
        self.min_obs = self.period

    def _estimate(self) -> None:
        pass                                   # nothing to estimate

    def _extrapolate(self, horizon: int) -> np.ndarray:
        season = self.y_[-self.period:]
        return np.array([season[i % self.period] for i in range(horizon)])

    def fitted(self) -> np.ndarray:
        if self.y_ is None:
            raise RuntimeError("call fit() before fitted() on SeasonalNaive")
        out = np.full_like(self.y_, np.nan)
        out[self.period:] = self.y_[:-self.period]
        return out


# ----------------------------------------------------------------------
# Task 4: backtesting, through the shared mixin
# ----------------------------------------------------------------------
def compare_models(series: ArrayLike, builders: dict[str, Callable[[], Forecaster]],
                   first_origin: int = 3) -> pd.DataFrame:
    """Walk-forward accuracy of several models, best first.

    ``first_origin`` is a zero-based index, so the brief's "days 4-10" is
    ``first_origin=3``. The backtesting itself comes from ``WalkForwardMixin``,
    which every model inherits -- nothing is reimplemented here.
    """
    if not builders:
        raise ValueError("need at least one model to compare")
    rows = []
    for name, build in builders.items():
        suite = build().walk_forward_score(series, first_origin)
        rows.append(suite.as_row(name))
    return pd.DataFrame(rows).set_index("model").sort_values("MAE")


def tune_smoothing(series: ArrayLike, grid: ArrayLike | None = None,
                   first_origin: int = 3) -> tuple[float, pd.Series]:
    """Grid-search ``alpha`` on walk-forward MAE, via the shared helper."""
    candidates = (np.round(np.arange(0.1, 1.01, 0.05), 2) if grid is None
                  else np.asarray(grid, dtype=float).ravel())
    if candidates.size == 0:
        raise ValueError("the grid must not be empty")
    if np.any((candidates <= 0) | (candidates > 1)):
        raise ValueError("every alpha must be in (0, 1]")
    best, scores = grid_search(ExponentialSmoothing, candidates, series, first_origin)
    series_out = pd.Series(scores, name="MAE")
    series_out.index.name = "alpha"
    return best, series_out


# ----------------------------------------------------------------------
# Task 6: fleet sizing
# ----------------------------------------------------------------------
@dataclass(frozen=True)
class FleetSizer:
    """Turns a passenger forecast into a vehicle count.

    Parameters
    ----------
    seats : int
        Seats per vehicle; the brief says 14.
    trips : int
        One-way trips per vehicle per day; the brief says 8.
    buffer : float
        Spare capacity as a fraction; the brief says 15%.
    min_utilisation : float
        The lowest average load factor the association will accept before it
        would rather run one fewer vehicle.

    Notes
    -----
    Rounding is to **nearest**, not up, subject to a floor: a vehicle is only
    added if doing so leaves average utilisation at or above
    ``min_utilisation``. Rounding every route up regardless would put a second
    matatu on a route carrying 1.05 vehicles' worth of passengers, running it
    at 52% full all day, and fuel and driver costs make that worse than the
    occasional overflow.
    """

    seats: int = 14
    trips: int = 8
    buffer: float = 0.15
    min_utilisation: float = 0.60

    def __post_init__(self) -> None:
        if self.seats < 1 or self.trips < 1:
            raise ValueError("seats and trips must be at least 1")
        if not 0 <= self.buffer < 5:
            raise ValueError("buffer must be a fraction in [0, 5)")
        if not 0 < self.min_utilisation <= 1:
            raise ValueError("min_utilisation must be in (0, 1]")

    @property
    def daily_capacity(self) -> int:
        """Passengers one vehicle can carry in a day."""
        return self.seats * self.trips

    def vehicles(self, passengers: float) -> int:
        """Vehicles needed, rounded to nearest subject to the utilisation floor."""
        if passengers < 0:
            raise ValueError("passenger count cannot be negative")
        if passengers == 0:
            return 0
        needed = passengers * (1 + self.buffer) / self.daily_capacity
        rounded = max(1, int(np.floor(needed + 0.5)))
        if rounded > 1 and passengers / (rounded * self.daily_capacity) < \
                self.min_utilisation:
            rounded -= 1
        return max(1, rounded)

    def utilisation(self, passengers: float, vehicles: int) -> float:
        """Average load factor implied by a vehicle count."""
        if vehicles < 1:
            raise ValueError("vehicles must be at least 1")
        return passengers / (vehicles * self.daily_capacity)

    def plan(self, forecasts: dict[str, float]) -> pd.DataFrame:
        """A vehicle count and load factor for each route."""
        rows = []
        for route, passengers in forecasts.items():
            count = self.vehicles(passengers)
            rows.append({"route": route, "forecast passengers": passengers,
                         "with buffer": passengers * (1 + self.buffer),
                         "vehicles": count,
                         "utilisation": self.utilisation(passengers, max(count, 1))})
        return pd.DataFrame(rows).set_index("route")


def load_routes(names: Sequence[str] | None = None) -> list[Route]:
    """Build :class:`Route` objects from the brief's data."""
    wanted = list(ROUTE_DATA) if names is None else list(names)
    absent = [n for n in wanted if n not in ROUTE_DATA]
    if absent:
        raise ValueError(f"unknown route(s) {absent}; have {sorted(ROUTE_DATA)}")
    return [Route(n, ROUTE_DATA[n]["counts"], ROUTE_DATA[n]["fare"]) for n in wanted]


def simulate_weekly_pattern(days: int = 60, base: float = 50.0, seed: int = 1234,
                            multipliers: Sequence[float] | None = None,
                            noise: float = 0.07) -> np.ndarray:
    """Extension: synthetic daily demand with a weekly rhythm.

    Busier on Fridays and much quieter on Sundays, as the brief describes, so
    a seasonal model has a pattern to find.
    """
    n = int(days)
    if n < 1:
        raise ValueError("days must be at least 1")
    weights = np.asarray(
        (1.00, 1.03, 1.06, 1.12, 1.30, 0.93, 0.70) if multipliers is None
        else multipliers, dtype=float)
    if weights.size != 7:
        raise ValueError("multipliers needs exactly seven values")
    if base <= 0 or noise < 0:
        raise ValueError("base must be positive and noise non-negative")
    rng = np.random.default_rng(seed)
    return np.round(base * weights[np.arange(n) % 7] * rng.normal(1.0, noise, n), 0)
