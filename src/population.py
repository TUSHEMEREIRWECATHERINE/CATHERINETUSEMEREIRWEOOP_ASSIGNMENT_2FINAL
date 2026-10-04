"""Mini-Project 1: UBOS District Population Forecaster.

A district planning unit needs five-year forecasts in order to budget primary
school classrooms. This module supplies the domain objects; the notebook runs
the analysis and interprets it.

Classes
-------
:class:`DistrictPopulation`
    One district's series. Behaves like a read-only mapping of year to
    population, so ``series[2020]`` and ``for year, people in series`` both
    work.
:class:`PolyfitTrend`, :class:`CompoundGrowth`, :class:`GoldenRatioProjection`
    The three models the brief specifies, each filling in the ``_estimate``
    and ``_extrapolate`` hooks of :class:`~src.forecasting.Forecaster`.
:class:`ValidationBench`
    Holds the train/test split and scores a set of models against it.
:class:`SchoolCapacityModel`
    Turns forecast growth into a classroom requirement.
"""
from __future__ import annotations

import statistics
from dataclasses import dataclass
from typing import Iterator

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike

from src.forecasting import EnsembleForecaster, Forecaster
from src.metrics import MetricSuite
from src.mixins import SeriesValidationMixin

#: Years covered by the illustrative series.
YEARS: tuple[int, ...] = tuple(range(2015, 2025))

#: Populations in thousands. Kampala, Wakiso and Gulu are the brief's figures.
#: Masaka and Lira are the two additional districts required by task 1: both
#: are regional service centres rather than suburbs of Kampala, which gives
#: the comparison a slower-growing counterweight to the central belt.
POPULATIONS: dict[str, tuple[float, ...]] = {
    "Kampala": (1200, 1250, 1300, 1350, 1420, 1500, 1580, 1650, 1720, 1800),
    "Wakiso": (950, 1000, 1070, 1150, 1220, 1300, 1390, 1480, 1570, 1670),
    "Gulu": (320, 330, 345, 360, 375, 390, 410, 430, 455, 480),
    "Masaka": (290, 299, 306, 317, 324, 335, 345, 353, 365, 375),
    "Lira": (410, 425, 444, 459, 479, 496, 518, 536, 559, 580),
}

#: Final year of the training window; 2022-2024 is held out.
TRAIN_UNTIL = 2021

#: Horizon the planning unit asks for: 2025-2029.
PLANNING_HORIZON = 5


class DistrictPopulation(SeriesValidationMixin):
    """A district's population series, in thousands of people.

    Inherits only the validation mixin: a container needs the input rules but
    none of the forecasting machinery.

    Parameters
    ----------
    district : str
        District name, a non-empty string.
    years : array-like of int
        Calendar years, strictly increasing.
    thousands : array-like of float
        Population in thousands; finite, non-negative, same length as ``years``.

    Raises
    ------
    ValueError
        On empty input, mismatched lengths, negative or non-finite values, or
        years that do not increase.
    """

    min_obs = 1

    def __init__(self, district: str, years: ArrayLike, thousands: ArrayLike) -> None:
        if not isinstance(district, str) or not district.strip():
            raise ValueError("district must be a non-empty string")
        values = self.clean(thousands, "thousands", self.min_obs)
        periods = np.asarray(years, dtype=int).ravel()
        if periods.size != values.size:
            raise ValueError(
                f"years ({periods.size}) and thousands ({values.size}) differ in length")
        if np.any(values < 0):
            raise ValueError("population cannot be negative")
        if periods.size > 1 and np.any(np.diff(periods) <= 0):
            raise ValueError("years must be strictly increasing")
        self.district = district.strip()
        self.years = periods
        self.thousands = values

    # ----- container behaviour -----------------------------------------
    def __repr__(self) -> str:
        return (f"<DistrictPopulation {self.district}: {self.years[0]}-"
                f"{self.years[-1]}, {len(self)} points, "
                f"last {self.thousands[-1]:,.0f}k>")

    def __len__(self) -> int:
        return int(self.years.size)

    def __getitem__(self, year: int) -> float:
        """``series[2020]`` -- the population recorded in that year."""
        match = np.flatnonzero(self.years == year)
        if match.size == 0:
            raise KeyError(f"{year} is outside {self.years[0]}-{self.years[-1]}")
        return float(self.thousands[match[0]])

    def __iter__(self) -> Iterator[tuple[int, float]]:
        """Iterate as ``(year, population)`` pairs."""
        return zip(self.years.tolist(), self.thousands.tolist())

    def __contains__(self, year: object) -> bool:
        return bool(np.any(self.years == year))

    # ----- task 2: two routes to the same statistics --------------------
    def summary_stdlib(self) -> dict[str, float]:
        """Mean, median, variance and standard deviation via ``statistics``.

        ``statistics.variance`` is the sample variance: it divides by n-1.
        """
        values = self.thousands.tolist()
        if len(values) < 2:
            raise ValueError("variance needs at least two observations")
        return {"mean": statistics.mean(values), "median": statistics.median(values),
                "variance": statistics.variance(values),
                "stdev": statistics.stdev(values)}

    def summary_numpy(self, ddof: int = 0) -> dict[str, float]:
        """The same four statistics via NumPy.

        ``np.var`` divides by ``n - ddof`` and ``ddof`` defaults to 0, giving
        the population variance. ``ddof=1`` reproduces :meth:`summary_stdlib`.
        """
        v = self.thousands
        return {"mean": float(np.mean(v)), "median": float(np.median(v)),
                "variance": float(np.var(v, ddof=ddof)),
                "stdev": float(np.std(v, ddof=ddof))}

    # ----- task 3: growth ----------------------------------------------
    def annual_changes(self) -> np.ndarray:
        """Year-on-year growth as fractions; one shorter than the series."""
        if len(self) < 2:
            raise ValueError("need at least two years to measure change")
        if np.any(self.thousands[:-1] == 0):
            raise ValueError("growth is undefined following a zero")
        return self.thousands[1:] / self.thousands[:-1] - 1

    def cagr(self) -> float:
        """Compound annual growth rate across the whole series."""
        span = int(self.years[-1] - self.years[0])
        if span <= 0:
            raise ValueError("CAGR needs at least two distinct years")
        if self.thousands[0] == 0:
            raise ValueError("CAGR is undefined from a zero starting value")
        return float((self.thousands[-1] / self.thousands[0]) ** (1 / span) - 1)

    # ----- task 5: the split --------------------------------------------
    def partition(self, train_until: int) -> tuple["DistrictPopulation",
                                                   "DistrictPopulation"]:
        """Return (train, test); ``train_until`` is the last training year."""
        in_train = self.years <= train_until
        if not in_train.any() or in_train.all():
            raise ValueError(
                f"a split at {train_until} leaves one side empty; the series "
                f"covers {self.years[0]}-{self.years[-1]}")
        return (DistrictPopulation(self.district, self.years[in_train],
                                   self.thousands[in_train]),
                DistrictPopulation(self.district, self.years[~in_train],
                                   self.thousands[~in_train]))


# ----------------------------------------------------------------------
# Task 4: three models, three sets of assumptions
# ----------------------------------------------------------------------
class PolyfitTrend(Forecaster):
    """Least-squares straight line through the series, via ``np.polyfit``.

    Assumes a district adds a constant *number* of people each year. It cannot
    curve, so against a compounding series it falls steadily behind.
    """

    min_obs = 2
    label = "polyfit-trend"

    def __init__(self) -> None:
        super().__init__()
        self.gradient_: float | None = None
        self.offset_: float | None = None

    def _estimate(self) -> None:
        self.gradient_, self.offset_ = (float(c) for c in np.polyfit(self.t_, self.y_, 1))

    def _extrapolate(self, horizon: int) -> np.ndarray:
        ahead = self.t_[-1] + np.arange(1, horizon + 1, dtype=float)
        return self.gradient_ * ahead + self.offset_

    def fitted(self) -> np.ndarray:
        if self.y_ is None:
            raise RuntimeError("call fit() before fitted() on PolyfitTrend")
        return self.gradient_ * self.t_ + self.offset_


class CompoundGrowth(Forecaster):
    """Constant *percentage* growth, implied by the first and last observations.

    Populations compound rather than add, so this is the natural rival to the
    straight line.
    """

    min_obs = 2
    label = "compound-growth"

    def __init__(self) -> None:
        super().__init__()
        self.rate_: float | None = None

    def _estimate(self) -> None:
        span = float(self.t_[-1] - self.t_[0])
        if span <= 0:
            raise ValueError("compound growth needs a positive time span")
        if self.y_[0] <= 0:
            raise ValueError("compound growth needs a positive first observation")
        self.rate_ = float((self.y_[-1] / self.y_[0]) ** (1 / span) - 1)

    def _extrapolate(self, horizon: int) -> np.ndarray:
        return self.y_[-1] * (1 + self.rate_) ** np.arange(1, horizon + 1, dtype=float)

    def fitted(self) -> np.ndarray:
        if self.y_ is None:
            raise RuntimeError("call fit() before fitted() on CompoundGrowth")
        return self.y_[0] * (1 + self.rate_) ** (self.t_ - self.t_[0])


class GoldenRatioProjection(Forecaster):
    """Scales the last observation by successive Fibonacci ratios.

    Inherited from the previous cohort of this assignment and kept so that it
    can be assessed rather than assumed. ``from_term`` sets where in the
    sequence the ratios start; this build starts at term 2, so the early,
    still-unsettled ratios (2.0, 1.5, 1.667, 1.6) are visible in the first few
    forecast steps before they settle on the golden ratio.
    """

    min_obs = 1
    label = "golden-ratio"

    def __init__(self, from_term: int = 2) -> None:
        super().__init__()
        self.from_term = self.whole_number(from_term, "from_term")
        self.ratios_: np.ndarray | None = None

    def _estimate(self) -> None:
        self.ratios_ = None  # depends on the horizon, so computed on demand

    def _extrapolate(self, horizon: int) -> np.ndarray:
        sequence = fibonacci(self.from_term + horizon + 2)
        window = sequence[self.from_term: self.from_term + horizon + 1]
        self.ratios_ = window[1:] / window[:-1]
        return self.y_[-1] * np.cumprod(self.ratios_)


def fibonacci(count: int) -> np.ndarray:
    """The first ``count`` Fibonacci numbers as floats: 1, 1, 2, 3, 5, ..."""
    if isinstance(count, bool) or not isinstance(count, (int, np.integer)) or count < 1:
        raise ValueError(f"count must be a positive integer, got {count!r}")
    out = np.empty(int(count), dtype=float)
    previous, current = 1.0, 1.0
    for i in range(int(count)):
        out[i], previous, current = previous, current, previous + current
    return out


#: The models entered into the comparison, as zero-argument builders.
MODEL_BUILDERS: dict[str, type[Forecaster]] = {
    PolyfitTrend.label: PolyfitTrend,
    CompoundGrowth.label: CompoundGrowth,
    GoldenRatioProjection.label: GoldenRatioProjection,
}


# ----------------------------------------------------------------------
# Tasks 5-6: validate, then forecast
# ----------------------------------------------------------------------
class ValidationBench:
    """Scores a set of models on a held-out window.

    Parameters
    ----------
    train_until : int
        Last year kept for training.
    builders : mapping of str to Forecaster subclass, optional
        What to compare. Defaults to the three models the brief specifies.
    """

    def __init__(self, train_until: int = TRAIN_UNTIL,
                 builders: dict[str, type[Forecaster]] | None = None) -> None:
        self.train_until = int(train_until)
        self.builders = dict(MODEL_BUILDERS if builders is None else builders)
        if not self.builders:
            raise ValueError("need at least one model to compare")

    def __repr__(self) -> str:
        return (f"ValidationBench(train_until={self.train_until}, "
                f"models={sorted(self.builders)})")

    def __len__(self) -> int:
        return len(self.builders)

    def scores(self, series: DistrictPopulation) -> dict[str, MetricSuite]:
        """Out-of-sample accuracy of every model for one district."""
        train, test = series.partition(self.train_until)
        out = {}
        for name, builder in self.builders.items():
            model = builder().fit(train.thousands, train.years.astype(float))
            out[name] = model.score(test.thousands)
        return out

    def table(self, series: DistrictPopulation) -> pd.DataFrame:
        """Comparison table for one district, most accurate first."""
        rows = [suite.as_row(name) for name, suite in self.scores(series).items()]
        return (pd.DataFrame(rows).set_index("model")
                .sort_values("MAPE_%"))

    def winner(self, series: DistrictPopulation) -> str:
        """Name of the model with the lowest MAPE for one district."""
        scores = self.scores(series)
        return min(scores, key=lambda name: scores[name].mape)

    def refit_winner(self, series: DistrictPopulation) -> Forecaster:
        """The winning model, refitted on the district's complete series."""
        builder = self.builders[self.winner(series)]
        return builder().fit(series.thousands, series.years.astype(float))


def projection(series: DistrictPopulation, model: Forecaster,
               horizon: int = PLANNING_HORIZON) -> pd.Series:
    """Forecast values indexed by calendar year."""
    years = series.years[-1] + np.arange(1, horizon + 1)
    return pd.Series(model.predict(horizon), index=years, name=series.district)


def spread_check(series: DistrictPopulation, forecast: ArrayLike) -> dict[str, float]:
    """Task 6: the observed variance against the forecast variance.

    A fitted curve has had the year-to-year scatter taken out of it, so a
    smaller forecast variance is a statement about the estimator, not a
    prediction that the future will be steadier.
    """
    values = np.asarray(forecast, dtype=float).ravel()
    if values.size < 2:
        raise ValueError("need at least two forecast values to take a variance")
    observed = float(np.var(series.thousands, ddof=1))
    projected = float(np.var(values, ddof=1))
    return {"observed_variance": observed, "forecast_variance": projected,
            "ratio": projected / observed if observed else float("inf")}


# ----------------------------------------------------------------------
# Extension: prediction intervals from resampled residuals
# ----------------------------------------------------------------------
def residual_bootstrap(model: Forecaster, horizon: int = PLANNING_HORIZON,
                       draws: int = 1000, coverage: float = 0.95,
                       seed: int = 1234) -> pd.DataFrame:
    """Prediction intervals by resampling the model's own training residuals.

    A point forecast carries no uncertainty, so the spread is reconstructed
    from how far the fitted curve missed the observations it was trained on.

    Parameters
    ----------
    model : Forecaster
        An already-fitted model.
    horizon : int
        How many steps to forecast.
    draws : int
        Bootstrap resamples; the brief asks for at least 1,000.
    coverage : float
        Interval coverage, strictly between 0 and 1.
    seed : int
        Passed to ``np.random.default_rng`` so the interval is reproducible.

    Returns
    -------
    pandas.DataFrame
        Columns ``point``, ``lower`` and ``upper``, one row per step.
    """
    steps = model.whole_number(horizon, "horizon")
    n_draws = model.whole_number(draws, "draws")
    if not 0 < coverage < 1:
        raise ValueError("coverage must be strictly between 0 and 1")
    centre = model.predict(steps)
    errors = model.residuals()
    errors = errors[np.isfinite(errors)]
    if errors.size == 0:
        raise ValueError("the model has no finite residuals to resample")
    rng = np.random.default_rng(seed)
    simulated = centre + rng.choice(errors, size=(n_draws, steps), replace=True)
    edge = (1 - coverage) / 2 * 100
    return pd.DataFrame({"point": centre,
                         "lower": np.percentile(simulated, edge, axis=0),
                         "upper": np.percentile(simulated, 100 - edge, axis=0)})


# ----------------------------------------------------------------------
# Task 8: the planning output
# ----------------------------------------------------------------------
@dataclass(frozen=True)
class SchoolCapacityModel:
    """Converts population growth into a primary-classroom requirement.

    Frozen, so a set of planning assumptions cannot be edited halfway through
    an analysis; a different scenario means a new, clearly-named object.

    Parameters
    ----------
    primary_share : float
        Share of the population of primary-school age (the brief says 18%).
    pupils_per_room : int
        Classroom capacity (the brief says 53).
    """

    primary_share: float = 0.18
    pupils_per_room: int = 53

    def __post_init__(self) -> None:
        if not 0 < self.primary_share <= 1:
            raise ValueError("primary_share must be in (0, 1]")
        if self.pupils_per_room < 1:
            raise ValueError("pupils_per_room must be at least 1")

    def enrolment(self, thousands: float) -> float:
        """Primary-age pupils implied by a population given in thousands."""
        if thousands < 0:
            raise ValueError("population cannot be negative")
        return thousands * 1_000 * self.primary_share

    def rooms_for(self, thousands: float) -> int:
        """Rooms needed to seat the whole of that population, rounded up."""
        return int(np.ceil(self.enrolment(thousands) / self.pupils_per_room))

    def rooms_to_add(self, baseline: float, projected: float) -> int:
        """Extra rooms needed for the growth between two population levels."""
        increase = projected - baseline
        return 0 if increase <= 0 else int(
            np.ceil(self.enrolment(increase) / self.pupils_per_room))


def load_series(districts: list[str] | None = None) -> list[DistrictPopulation]:
    """Build :class:`DistrictPopulation` objects from the illustrative data."""
    wanted = list(POPULATIONS) if districts is None else list(districts)
    missing = [d for d in wanted if d not in POPULATIONS]
    if missing:
        raise ValueError(f"unknown district(s): {missing}; have {sorted(POPULATIONS)}")
    return [DistrictPopulation(d, YEARS, POPULATIONS[d]) for d in wanted]


__all__ = [
    "YEARS", "POPULATIONS", "TRAIN_UNTIL", "PLANNING_HORIZON", "MODEL_BUILDERS",
    "DistrictPopulation", "PolyfitTrend", "CompoundGrowth", "GoldenRatioProjection",
    "ValidationBench", "SchoolCapacityModel", "EnsembleForecaster",
    "fibonacci", "load_series", "projection", "residual_bootstrap", "spread_check",
]
