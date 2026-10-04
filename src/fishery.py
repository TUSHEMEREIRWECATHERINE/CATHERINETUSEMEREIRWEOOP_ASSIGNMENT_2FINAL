"""Mini-Project 3: Lake Victoria Fish Stock & Export Risk Model.

A fish-export cooperative in Jinja wants to know whether its harvesting rate
is sustainable and how risky its revenue is. The previous version of this
question modelled the stock with Fibonacci numbers and flagged risk when
revenue variance exceeded 50,000; both are replaced here, and the notebook
shows why neither was defensible.

Classes
-------
:class:`HarvestPolicy`
    How hard, and when, the fishery is worked. A separate object because the
    closed-season extension is a change of *policy*, not of biology.
:class:`FishStock`
    Discrete logistic growth, told what to do by a policy.
:class:`PriceWalk`
    A seeded random walk for the weekly export price.
:class:`RiskProfile`
    CV bands, Value-at-Risk and downside semi-deviation.
:class:`FisheryCase`
    Composes a stock, a price process and a risk rule into one scenario.
"""
from __future__ import annotations

import statistics
from dataclasses import dataclass, field
from typing import Iterator, Sequence

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike

from src.mixins import SeriesValidationMixin

__all__ = ["KG_PER_TONNE", "FisheryCase", "FishStock", "HarvestPolicy", "PriceWalk",
           "RiskProfile", "SimulationRun", "fibonacci_series"]

#: The stock is in tonnes, the price is per kilogram.
KG_PER_TONNE = 1_000

#: Risk bands on the coefficient of variation. Export earnings from soft
#: commodities are usually reported with a CV in the 0.15-0.25 range, so a
#: "low" verdict here means genuinely steadier than a typical exporter and
#: "high" means materially more exposed.
LOW_CV, HIGH_CV = 0.12, 0.28


def fibonacci_series(count: int = 15) -> np.ndarray:
    """``count`` Fibonacci numbers, kept only so the baseline can be criticised."""
    if isinstance(count, bool) or not isinstance(count, int) or count < 1:
        raise ValueError(f"count must be a positive integer, got {count!r}")
    out = np.empty(count, dtype=float)
    first, second = 1.0, 1.0
    for i in range(count):
        out[i], first, second = first, second, first + second
    return out


@dataclass(frozen=True)
class HarvestPolicy:
    """How hard the fishery is worked, and when it is shut.

    Separating the policy from the biology is what makes the closed-season
    extension a one-line change: the stock dynamics are a fact about the lake,
    whereas the harvest rate and the closure are decisions.

    Parameters
    ----------
    rate : float
        Fraction of the standing stock taken in each open week.
    closed_weeks : tuple of int
        Zero-based week numbers with no harvesting. The default closes weeks
        0-7, placing the eight-week closure at the start of the calendar year
        over the short-rains spawning period.
    """

    rate: float = 0.10
    closed_weeks: tuple[int, ...] = ()

    def __post_init__(self) -> None:
        if not np.isfinite(self.rate) or not 0 <= self.rate < 1:
            raise ValueError("rate must be in [0, 1)")
        if any(w < 0 for w in self.closed_weeks):
            raise ValueError("closed weeks cannot be negative")

    @classmethod
    def with_closed_season(cls, rate: float = 0.10, start: int = 0,
                           length: int = 8) -> "HarvestPolicy":
        """A policy that shuts ``length`` consecutive weeks from ``start``."""
        if length < 0:
            raise ValueError("length cannot be negative")
        return cls(rate=rate, closed_weeks=tuple(range(start, start + length)))

    def open_mask(self, weeks: int) -> np.ndarray:
        """Boolean mask: is the fishery open in each of ``weeks`` weeks?"""
        if weeks < 1:
            raise ValueError("weeks must be at least 1")
        if any(w >= weeks for w in self.closed_weeks):
            raise ValueError(f"closed weeks must all lie below {weeks}")
        mask = np.ones(int(weeks), dtype=bool)
        if self.closed_weeks:
            mask[list(self.closed_weeks)] = False
        return mask

    def rate_in(self, week: int) -> float:
        """The harvest rate applied in a given week."""
        return 0.0 if week in self.closed_weeks else self.rate

    def __len__(self) -> int:
        """How many weeks the fishery is shut."""
        return len(self.closed_weeks)

    def __str__(self) -> str:
        if not self.closed_weeks:
            return f"h = {self.rate:.2f}, open all year"
        return (f"h = {self.rate:.2f}, closed weeks "
                f"{min(self.closed_weeks)}-{max(self.closed_weeks)}")


@dataclass
class SimulationRun:
    """The outcome of one simulated year."""

    biomass: np.ndarray
    catch: np.ndarray
    policy: HarvestPolicy

    def __len__(self) -> int:
        return int(self.catch.size)

    def __iter__(self) -> Iterator[tuple[float, float]]:
        """``(biomass at the start of the week, catch taken)`` pairs."""
        return zip(self.biomass[:-1].tolist(), self.catch.tolist())

    def __repr__(self) -> str:
        return (f"SimulationRun({self.policy}, weeks={len(self)}, "
                f"final={self.closing_biomass:,.0f}t, "
                f"landed={self.landed:,.0f}t)")

    @property
    def closing_biomass(self) -> float:
        return float(self.biomass[-1])

    @property
    def landed(self) -> float:
        """Everything caught over the run, in tonnes."""
        return float(self.catch.sum())

    def revenue(self, price_per_kg: ArrayLike) -> np.ndarray:
        """Weekly revenue in UGX: tonnes x 1,000 kg x price per kg."""
        price = np.asarray(price_per_kg, dtype=float).ravel()
        if price.size != len(self):
            raise ValueError(
                f"need {len(self)} prices, one per week, got {price.size}")
        return self.catch * KG_PER_TONNE * price


class FishStock(SeriesValidationMixin):
    """Discrete logistic growth with harvesting.

    .. math:: N_{t+1} = N_t + r N_t (1 - N_t/K) - h N_t

    The bracket is what the Fibonacci baseline lacks: as the biomass
    approaches the carrying capacity it goes to zero and growth stops.

    Parameters
    ----------
    r : float
        Intrinsic weekly growth rate; the brief gives 0.4.
    K : float
        Carrying capacity in tonnes; the brief gives 10,000.
    N0 : float
        Starting biomass in tonnes; the brief gives 4,000.
    """

    def __init__(self, r: float = 0.4, K: float = 10_000.0,
                 N0: float = 4_000.0) -> None:
        for label, value in (("r", r), ("K", K), ("N0", N0)):
            if not np.isfinite(value) or value <= 0:
                raise ValueError(f"{label} must be positive and finite")
        if N0 > K:
            raise ValueError(f"N0 ({N0:,.0f}) cannot exceed K ({K:,.0f})")
        self.r, self.K, self.N0 = float(r), float(K), float(N0)

    def __repr__(self) -> str:
        return f"FishStock(r={self.r}, K={self.K:,.0f}, N0={self.N0:,.0f})"

    # ----- reference points ---------------------------------------------
    @property
    def msy(self) -> float:
        """Maximum sustainable yield, rK/4 tonnes per week."""
        return self.r * self.K / 4

    @property
    def msy_rate(self) -> float:
        """The harvest rate achieving MSY, h = r/2."""
        return self.r / 2

    def settles_at(self, rate: float) -> float:
        """Equilibrium biomass under a constant rate: ``K(1 - h/r)``."""
        return max(0.0, self.K * (1 - float(rate) / self.r))

    def sustainable_yield(self, rate: float) -> float:
        """Catch per week at that equilibrium, ``h N*``."""
        return float(rate) * self.settles_at(rate)

    # ----- simulation ------------------------------------------------------
    def advance(self, biomass: float, rate: float) -> float:
        """One week of growth, net of harvest."""
        if biomass < 0:
            raise ValueError("biomass cannot be negative")
        grown = biomass + self.r * biomass * (1 - biomass / self.K) - rate * biomass
        return max(0.0, grown)

    def run(self, policy: HarvestPolicy, weeks: int = 52,
            start: float | None = None) -> SimulationRun:
        """Simulate ``weeks`` weeks under ``policy``."""
        n = self.whole_number(weeks, "weeks")
        is_open = policy.open_mask(n)
        biomass = np.empty(n + 1, dtype=float)
        catch = np.empty(n, dtype=float)
        biomass[0] = self.N0 if start is None else float(start)
        if biomass[0] < 0:
            raise ValueError("starting biomass cannot be negative")
        for week in range(n):
            rate = policy.rate if is_open[week] else 0.0
            catch[week] = rate * biomass[week]
            biomass[week + 1] = self.advance(biomass[week], rate)
        return SimulationRun(biomass, catch, policy)


class PriceWalk(SeriesValidationMixin):
    """Weekly export price in UGX/kg, as a bounded random walk.

    The bounds are applied by **clipping**. That is the simplest reading of the
    brief, and it has a side effect worth measuring rather than hiding: a
    clipped walk accumulates probability mass exactly on the boundary, so over
    a long run it spends an implausible amount of time sitting at the floor.
    :meth:`boundary_time` reports how often that happens, so the notebook can
    judge whether it matters at this horizon.

    Parameters
    ----------
    start : float
        Opening price; the brief gives 12,000.
    floor, ceiling : float
        Price bounds; the brief suggests 9,000 and 16,000.
    weekly_sd : float
        Standard deviation of the weekly step, in UGX.
    seed : int
        Seed for ``np.random.default_rng``.
    """

    def __init__(self, start: float = 12_000.0, floor: float = 9_000.0,
                 ceiling: float = 16_000.0, weekly_sd: float = 350.0,
                 seed: int = 1234) -> None:
        if not floor < ceiling:
            raise ValueError(f"need floor < ceiling, got {floor} and {ceiling}")
        if not floor <= start <= ceiling:
            raise ValueError(f"start {start} must lie in [{floor}, {ceiling}]")
        if weekly_sd <= 0 or not np.isfinite(weekly_sd):
            raise ValueError("weekly_sd must be positive and finite")
        self.start, self.floor, self.ceiling = float(start), float(floor), float(ceiling)
        self.weekly_sd, self.seed = float(weekly_sd), int(seed)

    def __repr__(self) -> str:
        return (f"PriceWalk(start={self.start:,.0f}, "
                f"[{self.floor:,.0f}, {self.ceiling:,.0f}], "
                f"sd={self.weekly_sd:,.0f}, seed={self.seed})")

    def simulate(self, weeks: int = 52, paths: int = 1,
                 seed: int | None = None) -> np.ndarray:
        """``paths`` price paths of ``weeks`` steps, shape ``(paths, weeks)``."""
        n = self.whole_number(weeks, "weeks")
        count = self.whole_number(paths, "paths")
        rng = np.random.default_rng(self.seed if seed is None else seed)
        steps = rng.normal(0.0, self.weekly_sd, size=(count, n))
        return np.clip(self.start + np.cumsum(steps, axis=1), self.floor, self.ceiling)

    def one(self, weeks: int = 52, seed: int | None = None) -> np.ndarray:
        return self.simulate(weeks, 1, seed)[0]

    def boundary_time(self, weeks: int = 52, paths: int = 500,
                      seed: int | None = None) -> float:
        """Share of simulated weeks spent pinned to a bound.

        The honest diagnostic for a clipped walk: if this is near zero the
        clipping is harmless at this horizon, and if it is large the bounded
        process is being badly distorted.
        """
        values = self.simulate(weeks, paths, seed)
        pinned = np.isclose(values, self.floor) | np.isclose(values, self.ceiling)
        return float(pinned.mean())


@dataclass(frozen=True)
class RiskProfile:
    """Classifies revenue risk, and measures the downside.

    Why the coefficient of variation rather than the variance? Revenue is
    measured in UGX, so its variance is in **UGX squared** -- a quantity with
    no interpretation and no natural scale. A threshold such as
    "variance > 50,000" therefore says nothing: against weekly revenues in the
    hundreds of millions it corresponds to a standard deviation of about
    UGX 224, and re-expressing the same revenue in thousands of shillings
    changes the number by a factor of a million while the risk is identical.
    The CV divides the standard deviation by the mean, so the units cancel.

    Parameters
    ----------
    low_above, high_above : float
        CV thresholds separating the three bands.
    """

    low_above: float = LOW_CV
    high_above: float = HIGH_CV

    def __post_init__(self) -> None:
        if not 0 < self.low_above < self.high_above:
            raise ValueError("need 0 < low_above < high_above")

    @staticmethod
    def describe(revenue: ArrayLike) -> dict[str, float]:
        """Mean, median, variance, sd and CV, computed with ``statistics``."""
        data = [float(v) for v in np.asarray(revenue, dtype=float).ravel()]
        if len(data) < 2:
            raise ValueError("need at least two observations")
        mean, sd = statistics.mean(data), statistics.stdev(data)
        return {"mean": mean, "median": statistics.median(data),
                "variance": statistics.variance(data), "stdev": sd,
                "cv": sd / mean if mean else float("inf")}

    def band(self, cv: float) -> str:
        """Map a CV onto ``low`` / ``moderate`` / ``high``."""
        if cv < 0 or not np.isfinite(cv):
            raise ValueError("cv must be finite and non-negative")
        if cv < self.low_above:
            return "low"
        return "moderate" if cv < self.high_above else "high"

    def classify(self, revenue: ArrayLike) -> str:
        return self.band(self.describe(revenue)["cv"])

    @staticmethod
    def value_at_risk(outcomes: ArrayLike, alpha: float = 0.05) -> float:
        """The revenue level undershot only ``alpha`` of the time."""
        draws = np.asarray(outcomes, dtype=float).ravel()
        if draws.size < 2:
            raise ValueError("need at least two outcomes")
        if not 0 < alpha < 1:
            raise ValueError("alpha must be strictly between 0 and 1")
        return float(np.percentile(draws, alpha * 100))

    @staticmethod
    def semi_deviation(outcomes: ArrayLike) -> float:
        """Standard deviation of the *downside* only.

        Ordinary volatility punishes an unexpectedly good year exactly as hard
        as a bad one, which is not how a cooperative experiences risk. The
        semi-deviation measures scatter below the mean alone, so it answers
        the question that actually matters for a business with fixed costs.
        """
        draws = np.asarray(outcomes, dtype=float).ravel()
        if draws.size < 2:
            raise ValueError("need at least two outcomes")
        mean = draws.mean()
        below = draws[draws < mean]
        if below.size == 0:
            return 0.0
        return float(np.sqrt(np.mean((below - mean) ** 2)))

    def tail(self, outcomes: ArrayLike, alpha: float = 0.05) -> dict[str, float]:
        """Mean, VaR, the shortfall it implies, and the semi-deviation."""
        draws = np.asarray(outcomes, dtype=float).ravel()
        mean = float(draws.mean())
        var = self.value_at_risk(draws, alpha)
        return {"mean": mean, "var": var, "shortfall": mean - var,
                "shortfall_pct": 100 * (mean - var) / mean if mean else float("nan"),
                "semi_deviation": self.semi_deviation(draws),
                "downside_cv": self.semi_deviation(draws) / mean if mean else float("inf")}


@dataclass
class FisheryCase:
    """One stock, one price process and one risk rule, run together.

    Composition rather than inheritance: a scenario *has* a stock, a price
    model and a risk rule. None is a kind of the others, and each is useful on
    its own.
    """

    stock: FishStock = field(default_factory=FishStock)
    price: PriceWalk = field(default_factory=PriceWalk)
    risk: RiskProfile = field(default_factory=RiskProfile)

    def __repr__(self) -> str:
        return f"FisheryCase({self.stock!r}, {self.price!r})"

    def year(self, policy: HarvestPolicy, weeks: int = 52,
             prices: ArrayLike | None = None,
             start: float | None = None) -> dict[str, object]:
        """Simulate one year and score it."""
        run = self.stock.run(policy, weeks, start)
        path = (self.price.one(weeks) if prices is None
                else np.asarray(prices, dtype=float).ravel())
        revenue = run.revenue(path)
        summary = self.risk.describe(revenue)
        return {"run": run, "prices": path, "revenue": revenue,
                "summary": summary, "band": self.risk.band(summary["cv"])}

    def annual_revenue_paths(self, policy: HarvestPolicy, weeks: int = 52,
                             paths: int = 1_000, seed: int | None = None,
                             start: float | None = None) -> np.ndarray:
        """Annual revenue under ``paths`` independent price paths.

        The biomass path is deterministic given the policy, so only the price
        is resampled: that is where the revenue uncertainty lives.
        """
        run = self.stock.run(policy, weeks, start)
        prices = self.price.simulate(weeks, paths, seed)
        return (run.catch * KG_PER_TONNE) @ prices.T

    def rate_sweep(self, rates: Sequence[float], weeks: int = 52,
                   paths: int = 1_000, closed: Sequence[int] = ()) -> pd.DataFrame:
        """Compare harvest rates on stock, catch, revenue and risk."""
        if not len(rates):
            raise ValueError("need at least one rate")
        rows = []
        for rate in rates:
            policy = HarvestPolicy(float(rate), tuple(closed))
            outcome = self.year(policy, weeks)
            annual = self.annual_revenue_paths(policy, weeks, paths)
            tail = self.risk.tail(annual)
            rows.append({
                "h": float(rate),
                "closing_stock_t": outcome["run"].closing_biomass,
                "equilibrium_t": self.stock.settles_at(float(rate)),
                "landed_t": outcome["run"].landed,
                "sustainable_t_per_week": self.stock.sustainable_yield(float(rate)),
                "revenue_ugx": float(outcome["revenue"].sum()),
                "cv": outcome["summary"]["cv"],
                "band": outcome["band"],
                "mean_annual_ugx": tail["mean"],
                "var5_ugx": tail["var"],
                "semi_dev_ugx": tail["semi_deviation"],
            })
        return pd.DataFrame(rows).set_index("h")

    def multi_year(self, policy: HarvestPolicy, years: int = 5, weeks: int = 52,
                   seed: int | None = None) -> float:
        """Total revenue over several years, carrying the biomass forward."""
        if years < 1:
            raise ValueError("years must be at least 1")
        biomass, total = self.stock.N0, 0.0
        base_seed = self.price.seed if seed is None else int(seed)
        for year in range(int(years)):
            run = self.stock.run(policy, weeks, start=biomass)
            total += float(run.revenue(self.price.one(weeks, seed=base_seed + year)).sum())
            biomass = run.closing_biomass
        return total
